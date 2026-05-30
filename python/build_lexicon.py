"""Offline WLASL -> MediaPipe Holistic -> .pose lexicon builder.

Builds a word-level ASL pose lexicon that plugs straight into the
`spoken-to-signed` pipeline used by the server. For each gloss (English word)
in the WLASL dataset we download a candidate clip, crop it to the sign's
frame range, run MediaPipe Holistic to extract 543+ body/face/hand landmarks
per frame, and save a `.pose` file in the SAME format the bundled
fingerspelling lexicon uses (so the two concatenate cleanly).

The result is a directory like:

    lexicon_wlasl/
      index.csv
      ase/
        book.pose
        hello.pose
        ...

Point the server at it with  LEXICON_DIR=python/lexicon_wlasl  and any word in
the lexicon is signed with a real two-handed sign; everything else falls back
to fingerspelling automatically.

IMPORTANT: this script needs the *builder* virtualenv (Python 3.12 +
mediapipe 0.10.14 legacy Holistic). Run it with `.venv-b312/bin/python`, NOT
the server's `.venv` (Python 3.13, which has no legacy Holistic).

Usage examples
--------------
    # Build the 100 most-recorded glosses
    .venv-b312/bin/python build_lexicon.py --wlasl-json wlasl/WLASL_v0.3.json --num-glosses 100

    # Build a specific demo vocabulary
    .venv-b312/bin/python build_lexicon.py --wlasl-json wlasl/WLASL_v0.3.json \
        --glosses hello,book,help,family,learn

    # Use videos you already downloaded (named <video_id>.mp4) and skip network
    .venv-b312/bin/python build_lexicon.py --wlasl-json wlasl/WLASL_v0.3.json \
        --num-glosses 100 --videos-dir wlasl/videos --no-download
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# WLASL metadata                                                              #
# --------------------------------------------------------------------------- #
def load_wlasl(json_path, num_glosses=None, glosses_filter=None):
    """Return [(gloss, [instances])]. When glosses_filter is given, keep only
    those glosses; otherwise sort by number of instances (most-recorded first)
    and keep the top `num_glosses`.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = [(e["gloss"], e.get("instances", [])) for e in data]

    if glosses_filter:
        wanted = {g.strip().lower() for g in glosses_filter}
        entries = [(g, inst) for g, inst in entries if g.lower() in wanted]
    else:
        entries.sort(key=lambda gi: len(gi[1]), reverse=True)
        if num_glosses:
            entries = entries[:num_glosses]
    return entries


# --------------------------------------------------------------------------- #
# Video download                                                              #
# --------------------------------------------------------------------------- #
_FFMPEG_LOCATION = None


def _ffmpeg_location():
    global _FFMPEG_LOCATION
    if _FFMPEG_LOCATION is None:
        try:
            import imageio_ffmpeg

            _FFMPEG_LOCATION = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            _FFMPEG_LOCATION = ""
    return _FFMPEG_LOCATION


def resolve_video(instance, videos_dir, allow_download):
    """Return a local path to the instance's video, downloading if needed."""
    video_id = str(instance.get("video_id", "")).strip()
    os.makedirs(videos_dir, exist_ok=True)

    if video_id:
        for ext in (".mp4", ".mkv", ".webm", ".mov"):
            cached = os.path.join(videos_dir, video_id + ext)
            if os.path.isfile(cached) and os.path.getsize(cached) > 0:
                return cached

    if not allow_download:
        return None

    url = instance.get("url", "")
    if not url or url.lower().endswith(".swf"):
        return None

    out_tmpl = os.path.join(videos_dir, f"{video_id or '%(id)s'}.%(ext)s")
    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "mp4/bestvideo[ext=mp4]+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 2,
        "socket_timeout": 20,
        "ignoreerrors": True,
    }
    loc = _ffmpeg_location()
    if loc:
        ydl_opts["ffmpeg_location"] = loc

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return None
            path = ydl.prepare_filename(info)
            if os.path.isfile(path):
                return path
            base, _ = os.path.splitext(path)
            for ext in (".mp4", ".mkv", ".webm", ".mov"):
                if os.path.isfile(base + ext):
                    return base + ext
    except Exception as e:
        log(f"      download failed: {str(e)[:120]}")
    return None


# --------------------------------------------------------------------------- #
# Frame extraction                                                            #
# --------------------------------------------------------------------------- #
def read_clip_frames(video_path, frame_start, frame_end, bbox=None):
    """Read RGB frames for the sign's [frame_start, frame_end] range.

    WLASL frame indices are 1-based; frame_end == -1 means 'to the end'.
    Returns (frames, width, height, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []
    idx = 0
    start = max(0, (frame_start or 1) - 1)
    end = frame_end if (frame_end and frame_end > 0) else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start and (end is None or idx < end):
            if bbox:
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1 = max(0, min(int(x1), w - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y1 = max(0, min(int(y1), h - 1))
                y2 = max(y1 + 1, min(int(y2), h))
                frame = frame[y1:y2, x1:x2]
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if end is not None and idx >= end:
            break
    cap.release()

    if not frames:
        return [], 0, 0, fps
    h, w = frames[0].shape[:2]
    return frames, w, h, fps


# --------------------------------------------------------------------------- #
# MediaPipe Holistic -> Pose                                                  #
# --------------------------------------------------------------------------- #
def frames_to_pose(frames, width, height, fps):
    from pose_format.utils.holistic import load_holistic

    return load_holistic(
        frames,
        fps=fps,
        width=width,
        height=height,
        depth=0,
        additional_holistic_config={"refine_face_landmarks": True},
    )


def _hand_point_range(pose):
    """Return (start, end) point indices covering both hand components."""
    offset = 0
    lo, hi = None, None
    for c in pose.header.components:
        n = len(c.points)
        if c.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            lo = offset if lo is None else lo
            hi = offset + n
        offset += n
    return lo, hi


def hand_detection_fraction(pose):
    """Fraction of frames in which at least one hand landmark was detected."""
    lo, hi = _hand_point_range(pose)
    if lo is None:
        return 0.0
    conf = np.asarray(pose.body.confidence)  # (frames, people, points)
    if conf.ndim != 3 or conf.shape[0] == 0:
        return 0.0
    hand_conf = conf[:, :, lo:hi].sum(axis=(1, 2))
    return float((hand_conf > 0).mean())


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def build(args):
    entries = load_wlasl(
        args.wlasl_json,
        num_glosses=args.num_glosses,
        glosses_filter=args.glosses.split(",") if args.glosses else None,
    )
    if not entries:
        log("No glosses matched. Check --wlasl-json / --glosses / --num-glosses.")
        return 1

    out_dir = args.out
    pose_dir = os.path.join(out_dir, "ase")
    os.makedirs(pose_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.csv")

    rows = _read_index(index_path)
    built, failed = 0, 0
    log(f"Building lexicon for {len(entries)} glosses -> {out_dir}")

    for gi, (gloss, instances) in enumerate(entries, 1):
        rel_pose = f"ase/{gloss}.pose"
        abs_pose = os.path.join(out_dir, rel_pose)
        if os.path.isfile(abs_pose) and not args.overwrite:
            log(f"[{gi}/{len(entries)}] {gloss}: already built, skipping")
            rows.append(_row(rel_pose, gloss))
            built += 1
            continue

        log(f"[{gi}/{len(entries)}] {gloss}: {len(instances)} candidate clip(s)")
        success = False
        for ci, inst in enumerate(instances[: args.max_candidates]):
            video = resolve_video(inst, args.videos_dir, allow_download=not args.no_download)
            if not video:
                continue
            frames, w, h, fps = read_clip_frames(
                video,
                inst.get("frame_start", 1),
                inst.get("frame_end", -1),
                bbox=inst.get("bbox") if args.use_bbox else None,
            )
            if len(frames) < args.min_frames:
                log(f"      candidate {ci}: only {len(frames)} frames, skip")
                continue
            try:
                pose = frames_to_pose(frames, w, h, fps)
            except Exception as e:
                log(f"      candidate {ci}: holistic failed: {str(e)[:120]}")
                continue
            frac = hand_detection_fraction(pose)
            if frac < args.min_hand_fraction:
                log(f"      candidate {ci}: hands in {frac:.0%} of frames (<{args.min_hand_fraction:.0%}), skip")
                continue
            with open(abs_pose, "wb") as f:
                pose.write(f)
            log(f"      OK -> {rel_pose}  ({len(frames)} frames, hands {frac:.0%})")
            rows.append(_row(rel_pose, gloss))
            built += 1
            success = True
            break

        if not success:
            failed += 1
            log(f"      no usable clip for '{gloss}'")

    indexed = _write_index(index_path, rows)
    if indexed:
        log(f"\nDone. Built {built} gloss(es), {failed} failed. Index: {index_path}")
    else:
        log(f"\nDone. Built 0 usable glosses; no index written because no usable pose files were found.")
    log("Point the server at it:  LEXICON_DIR=" + os.path.abspath(out_dir))
    return 0


def _row(rel_pose, gloss):
    return {
        "path": rel_pose,
        "spoken_language": "en",
        "signed_language": "ase",
        "start": 0,
        "end": 0,
        "words": gloss,
        "glosses": gloss,
        "priority": 0,
    }


def _write_index(index_path, rows):
    directory = os.path.dirname(index_path)
    rows = [
        row for row in rows
        if row.get("path") and os.path.isfile(os.path.join(directory, row["path"]))
    ]
    if not rows:
        if os.path.exists(index_path):
            os.unlink(index_path)
        return 0

    # De-dup by path, keeping the last row so newly built entries replace
    # stale metadata for the same pose file.
    by_path = {}
    for row in rows:
        by_path[row["path"]] = row
    unique = list(by_path.values())
    fields = ["path", "spoken_language", "signed_language", "start", "end", "words", "glosses", "priority"]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(unique)
    return len(unique)


def _read_index(index_path):
    if not os.path.isfile(index_path):
        return []
    with open(index_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args():
    p = argparse.ArgumentParser(description="Build a WLASL -> .pose ASL lexicon for spoken-to-signed.")
    p.add_argument("--wlasl-json", required=True, help="Path to WLASL_v0.3.json")
    p.add_argument("--out", default="lexicon_wlasl", help="Output lexicon directory")
    p.add_argument("--videos-dir", default="wlasl/videos", help="Where to cache/find downloaded videos")
    p.add_argument("--num-glosses", type=int, default=None, help="Build the top-N most-recorded glosses")
    p.add_argument("--glosses", type=str, default=None, help="Comma-separated specific glosses to build")
    p.add_argument("--max-candidates", type=int, default=4, help="Max candidate clips to try per gloss")
    p.add_argument("--min-frames", type=int, default=8, help="Skip clips shorter than this many frames")
    p.add_argument("--min-hand-fraction", type=float, default=0.3,
                   help="Require hands detected in at least this fraction of frames")
    p.add_argument("--use-bbox", action="store_true", help="Crop frames to the WLASL signer bbox before extraction")
    p.add_argument("--no-download", action="store_true", help="Only use already-downloaded videos in --videos-dir")
    p.add_argument("--overwrite", action="store_true", help="Rebuild glosses even if a .pose already exists")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
