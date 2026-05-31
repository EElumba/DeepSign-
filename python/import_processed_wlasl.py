"""Import preprocessed WLASL MediaPipe landmarks into the local .pose lexicon.

This script is for datasets shaped like the Kaggle "MuteMotion: WLASL
MediaPipe Encoded" archive:

    wlasl_processed_source/archive/
      filtered_labels.txt
      landmarks_V3.npz
      WLASL_parsed_data.json

It avoids video downloads and MediaPipe processing by converting existing
landmark arrays directly into pose-format files. By default it imports only
missing glosses and preserves existing lexicon rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "deepsign-mpl"))

import numpy as np
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.utils.generic import reduce_holistic
from pose_format.utils.holistic import holistic_components


FIELDS = ["path", "spoken_language", "signed_language", "start", "end", "words", "glosses", "priority"]
V3_POINTS = 553
HOLISTIC_POINTS_WITH_WORLD = 586
SOURCE_RIGHT_HAND = slice(0, 21)
SOURCE_LEFT_HAND = slice(21, 42)
SOURCE_POSE = slice(42, 75)
SOURCE_FACE = slice(75, 553)
TARGET_POSE = slice(0, 33)
TARGET_FACE = slice(33, 511)
TARGET_LEFT_HAND = slice(511, 532)
TARGET_RIGHT_HAND = slice(532, 553)


def log(*args):
    print(*args, flush=True)


def slugify(gloss: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", gloss.strip().lower()).strip("_")
    return slug or "unknown"


def read_index(index_path: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        return []
    with index_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_index(index_path: Path, rows: list[dict[str, str]]) -> int:
    rows = [row for row in rows if row.get("path") and (index_path.parent / row["path"]).is_file()]
    by_path = {}
    for row in rows:
        by_path[row["path"]] = row

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(by_path.values())
    return len(by_path)


def row_for(rel_path: str, gloss: str) -> dict[str, str | int]:
    return {
        "path": rel_path,
        "spoken_language": "en",
        "signed_language": "ase",
        "start": 0,
        "end": 0,
        "words": gloss,
        "glosses": gloss,
        "priority": 0,
    }


def load_label_order(source_dir: Path, parsed: list[dict], requested: set[str] | None) -> list[str]:
    labels_path = source_dir / "filtered_labels.txt"
    if labels_path.is_file():
        labels = [line.strip().lower() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        counts = defaultdict(int)
        for sample in parsed:
            counts[sample["gloss"].lower()] += 1
        labels = [gloss for gloss, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]

    if requested is not None:
        return [gloss for gloss in labels if gloss in requested] + sorted(requested - set(labels))
    return labels


def available_samples(parsed: list[dict], landmark_keys: set[str]) -> dict[str, list[tuple[str, dict]]]:
    by_gloss = defaultdict(list)
    split_rank = {"train": 0, "val": 1, "test": 2}
    for idx, sample in enumerate(parsed):
        key = str(idx)
        if key not in landmark_keys:
            continue
        gloss = sample.get("gloss", "").strip().lower()
        if not gloss:
            continue
        by_gloss[gloss].append((key, sample))

    for gloss, samples in by_gloss.items():
        samples.sort(
            key=lambda item: (
                split_rank.get(item[1].get("split", ""), 9),
                -(item[1].get("frame_end", 0) if item[1].get("frame_end", 0) > 0 else 0),
                item[0],
            )
        )
    return by_gloss


def valid_points(arr: np.ndarray) -> np.ndarray:
    xy = arr[..., :2]
    return np.isfinite(arr).all(axis=-1) & (np.abs(xy).sum(axis=-1) > 0)


def hand_fraction(arr: np.ndarray) -> float:
    if arr.shape[1] < V3_POINTS:
        return 0.0
    hand_valid = valid_points(arr[:, :42, :])
    return float(hand_valid.any(axis=1).mean()) if len(hand_valid) else 0.0


def shoulder_width(arr: np.ndarray, width: int, height: int) -> float:
    pose = arr[:, SOURCE_POSE, :]
    conf = valid_points(pose)
    ok = conf[:, 11] & conf[:, 12]
    if not ok.any():
        return 0.0
    scaled = pose[:, :, :2].copy()
    scaled[..., 0] *= width
    scaled[..., 1] *= height
    distances = np.linalg.norm(scaled[ok, 11] - scaled[ok, 12], axis=1)
    return float(np.nanmean(distances))


def to_pose(arr: np.ndarray, fps: float, width: int, height: int, reduce: bool = True) -> Pose:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected landmarks shaped (frames, points, 3), got {arr.shape}")
    if arr.shape[1] != V3_POINTS:
        raise ValueError(f"only V3 arrays with {V3_POINTS} points are supported, got {arr.shape[1]}")

    frames = arr.shape[0]
    data = np.zeros((frames, 1, HOLISTIC_POINTS_WITH_WORLD, 3), dtype=np.float32)
    confidence = np.zeros((frames, 1, HOLISTIC_POINTS_WITH_WORLD), dtype=np.float32)

    converted = arr.astype(np.float32, copy=True)
    converted[..., 0] *= width
    converted[..., 1] *= height
    converted[~np.isfinite(converted)] = 0

    conf = valid_points(arr).astype(np.float32)
    # Kaggle V3 order is [Right Hand, Left Hand, Pose, Face]. pose-format's
    # holistic header expects [Pose, Face, Left Hand, Right Hand, Pose World].
    data[:, 0, TARGET_POSE, :] = converted[:, SOURCE_POSE, :]
    data[:, 0, TARGET_FACE, :] = converted[:, SOURCE_FACE, :]
    data[:, 0, TARGET_LEFT_HAND, :] = converted[:, SOURCE_LEFT_HAND, :]
    data[:, 0, TARGET_RIGHT_HAND, :] = converted[:, SOURCE_RIGHT_HAND, :]
    confidence[:, 0, TARGET_POSE] = conf[:, SOURCE_POSE]
    confidence[:, 0, TARGET_FACE] = conf[:, SOURCE_FACE]
    confidence[:, 0, TARGET_LEFT_HAND] = conf[:, SOURCE_LEFT_HAND]
    confidence[:, 0, TARGET_RIGHT_HAND] = conf[:, SOURCE_RIGHT_HAND]

    # The app's current lexicon uses refined holistic headers with a final
    # POSE_WORLD_LANDMARKS component. The Kaggle V3 file has the first 553
    # image-space points only, so keep world landmarks absent instead of
    # inventing coordinates.
    header = PoseHeader(
        version=0.2,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
        components=holistic_components("XYZC", additional_face_points=10),
    )
    pose = Pose(header, NumPyPoseBody(fps=fps, data=data, confidence=confidence))
    return reduce_holistic(pose) if reduce else pose


def build(args) -> int:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out)
    pose_dir = out_dir / "ase"
    index_path = out_dir / "index.csv"
    landmarks_path = source_dir / f"landmarks_{args.version}.npz"
    parsed_path = source_dir / "WLASL_parsed_data.json"

    if args.version != "V3":
        log("Only --version V3 is currently supported because V1/V2 use a reduced 180-point layout.")
        return 2
    if not landmarks_path.is_file() or not parsed_path.is_file():
        log(f"Missing required source files in {source_dir}")
        return 2

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    requested = set()
    if args.glosses:
        requested.update(g.strip().lower() for g in args.glosses.split(",") if g.strip())
    if args.glosses_file:
        requested.update(
            line.strip().lower()
            for line in Path(args.glosses_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    requested = requested or None
    rows = read_index(index_path)
    existing_glosses = {row.get("glosses", "").lower() for row in rows if row.get("glosses")}

    pose_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    failed = 0

    with np.load(landmarks_path) as landmarks:
        samples_by_gloss = available_samples(parsed, set(landmarks.keys()))
        labels = load_label_order(source_dir, parsed, requested)
        if args.num_glosses:
            labels = labels[: args.num_glosses]

        log(f"Importing {len(labels)} requested gloss(es) from {landmarks_path}")
        for idx, gloss in enumerate(labels, 1):
            rel_path = f"ase/{slugify(gloss)}.pose"
            abs_path = out_dir / rel_path
            if gloss in existing_glosses and not args.overwrite:
                skipped += 1
                log(f"[{idx}/{len(labels)}] {gloss}: already indexed, skipping")
                continue

            candidates = samples_by_gloss.get(gloss, [])[: args.max_candidates]
            if not candidates:
                failed += 1
                log(f"[{idx}/{len(labels)}] {gloss}: no landmark sample")
                continue

            success = False
            for key, _sample in candidates:
                arr = landmarks[key]
                if arr.shape[0] < args.min_frames:
                    continue
                hands = hand_fraction(arr)
                if hands < args.min_hand_fraction:
                    continue
                shoulders = shoulder_width(arr, args.width, args.height)
                if shoulders < args.min_shoulder_width:
                    log(f"      sample {key}: shoulder width {shoulders:.1f}px (<{args.min_shoulder_width}px), skip")
                    continue
                if not args.dry_run:
                    pose = to_pose(
                        arr,
                        fps=args.fps,
                        width=args.width,
                        height=args.height,
                        reduce=not args.keep_full_holistic,
                    )
                    with abs_path.open("wb") as f:
                        pose.write(f)
                    rows.append(row_for(rel_path, gloss))
                    existing_glosses.add(gloss)
                imported += 1
                success = True
                log(
                    f"[{idx}/{len(labels)}] {gloss}: OK sample {key} "
                    f"({arr.shape[0]} frames, hands {hands:.0%}, shoulders {shoulders:.0f}px)"
                )
                break

            if not success:
                failed += 1
                log(f"[{idx}/{len(labels)}] {gloss}: no usable candidate")

    if not args.dry_run:
        count = write_index(index_path, rows)
        log(f"\nDone. Imported {imported}, skipped {skipped}, failed {failed}. Index rows: {count}")
    else:
        log(f"\nDry run. Would import {imported}, skip {skipped}, fail {failed}.")
    return 0 if failed == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Convert processed WLASL MediaPipe landmarks into .pose lexicon files.")
    parser.add_argument("--source-dir", default="wlasl_processed_source/archive")
    parser.add_argument("--out", default="lexicon_wlasl")
    parser.add_argument("--version", default="V3", choices=["V3"])
    parser.add_argument("--num-glosses", type=int, default=None, help="Import the first N labels from filtered_labels.txt.")
    parser.add_argument("--glosses", default=None, help="Comma-separated glosses to import.")
    parser.add_argument("--glosses-file", default=None, help="Text file with one gloss per line to import.")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--min-hand-fraction", type=float, default=0.3)
    parser.add_argument("--min-shoulder-width", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--keep-full-holistic", action="store_true", help="Store all 586 points instead of reduced 178-point poses.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
