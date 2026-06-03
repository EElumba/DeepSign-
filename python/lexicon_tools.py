"""Developer tooling for DeepSign WLASL lexicon quality and aliases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lexicon_resolution import GlossResolver, gloss_slug, load_aliases, load_lexicon_glosses


DEFAULT_THRESHOLDS = {
    "min_frames": 8,
    "min_hand_frame_fraction": 0.30,
    "min_pose_frame_fraction": 0.30,
    "min_hand_landmark_coverage": 0.12,
    "min_pose_landmark_coverage": 0.30,
    "max_jump95": 180.0,
    "max_wrist_jump95": 220.0,
    "max_shoulder_cv": 0.35,
    "min_shoulder_width": 40.0,
}


def read_index(lexicon_dir: Path) -> list[dict[str, str]]:
    index_path = lexicon_dir / "index.csv"
    if not index_path.is_file():
        return []
    with index_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def component_slice(pose, component_name: str) -> slice | None:
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        if component.name == component_name:
            return slice(offset, offset + count)
        offset += count
    return None


def component_points(pose, component_name: str) -> list[str]:
    for component in pose.header.components:
        if component.name == component_name:
            return list(component.points)
    return []


def robust_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return default
    return float(np.percentile(values, percentile))


def safe_float(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 4)


def _slice_visibility(confidence: np.ndarray, point_slice: slice | None) -> tuple[float, float]:
    if point_slice is None or point_slice.stop <= point_slice.start:
        return 0.0, 0.0
    conf = confidence[:, 0, point_slice] > 0
    return float(conf.any(axis=1).mean()) if len(conf) else 0.0, float(conf.mean()) if conf.size else 0.0


def _jump95(data: np.ndarray, confidence: np.ndarray, point_slice: slice | None) -> float:
    if point_slice is None or data.shape[0] < 2:
        return 0.0
    visible = confidence[:, 0, point_slice] > 0
    both_visible = visible[1:] & visible[:-1]
    diffs = np.diff(data[:, 0, point_slice, :2], axis=0)
    jumps = np.linalg.norm(diffs, axis=-1)[both_visible]
    return robust_percentile(jumps, 95)


def _point_index(points: list[str], names: Iterable[str]) -> int | None:
    for name in names:
        if name in points:
            return points.index(name)
    return None


def _wrist_jump95(pose, hand_slices: list[slice | None], pose_slice: slice | None) -> float:
    wrists = []
    pose_points = component_points(pose, "POSE_LANDMARKS")
    if pose_slice is not None:
        for name in ("LEFT_WRIST", "RIGHT_WRIST"):
            idx = _point_index(pose_points, [name])
            if idx is not None:
                wrists.append(pose_slice.start + idx)

    for component_name, hand_slice in (
        ("LEFT_HAND_LANDMARKS", hand_slices[0]),
        ("RIGHT_HAND_LANDMARKS", hand_slices[1]),
    ):
        points = component_points(pose, component_name)
        idx = _point_index(points, ["WRIST", "0"])
        if hand_slice is not None and idx is not None:
            wrists.append(hand_slice.start + idx)

    values = []
    data = pose.body.data
    confidence = pose.body.confidence
    for wrist in wrists:
        if data.shape[0] < 2:
            continue
        ok = (confidence[1:, 0, wrist] > 0) & (confidence[:-1, 0, wrist] > 0)
        if ok.any():
            values.append(np.linalg.norm(np.diff(data[:, 0, wrist, :2], axis=0)[ok], axis=-1))
    if not values:
        return 0.0
    return robust_percentile(np.concatenate(values), 95)


def _shoulder_metrics(pose, pose_slice: slice | None) -> tuple[float, float]:
    if pose_slice is None:
        return 0.0, float("inf")
    points = component_points(pose, "POSE_LANDMARKS")
    left = _point_index(points, ["LEFT_SHOULDER", "11"])
    right = _point_index(points, ["RIGHT_SHOULDER", "12"])
    if left is None or right is None:
        return 0.0, float("inf")

    confidence = pose.body.confidence[:, 0, pose_slice]
    data = pose.body.data[:, 0, pose_slice, :2]
    ok = (confidence[:, left] > 0) & (confidence[:, right] > 0)
    if not ok.any():
        return 0.0, float("inf")

    distances = np.linalg.norm(data[ok, left] - data[ok, right], axis=1)
    mean = float(np.nanmean(distances)) if len(distances) else 0.0
    cv = float(np.nanstd(distances) / mean) if mean > 0 else float("inf")
    return mean, cv


def _quality_score(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[float, list[str]]:
    flags: list[str] = []
    penalty = 0.0

    if metrics["status"] != "ok":
        return 0.0, [metrics["status"]]

    if metrics["frames"] < thresholds["min_frames"]:
        flags.append("low_frame_count")
        penalty += (thresholds["min_frames"] - metrics["frames"]) * 5.0
    if metrics["hand_frame_fraction"] < thresholds["min_hand_frame_fraction"]:
        flags.append("unstable_or_missing_hands")
        penalty += (thresholds["min_hand_frame_fraction"] - metrics["hand_frame_fraction"]) * 100.0
    if metrics["pose_frame_fraction"] < thresholds["min_pose_frame_fraction"]:
        flags.append("incomplete_pose_visibility")
        penalty += (thresholds["min_pose_frame_fraction"] - metrics["pose_frame_fraction"]) * 80.0
    if metrics["hand_landmark_coverage"] < thresholds["min_hand_landmark_coverage"]:
        flags.append("incomplete_hand_landmarks")
        penalty += (thresholds["min_hand_landmark_coverage"] - metrics["hand_landmark_coverage"]) * 120.0
    if metrics["pose_landmark_coverage"] < thresholds["min_pose_landmark_coverage"]:
        flags.append("incomplete_pose_landmarks")
        penalty += (thresholds["min_pose_landmark_coverage"] - metrics["pose_landmark_coverage"]) * 80.0
    if metrics["jump95"] > thresholds["max_jump95"]:
        flags.append("excessive_pose_jumpiness")
        penalty += min(60.0, (metrics["jump95"] - thresholds["max_jump95"]) / 4.0)
    if metrics["wrist_jump95"] > thresholds["max_wrist_jump95"]:
        flags.append("unstable_wrist_tracking")
        penalty += min(50.0, (metrics["wrist_jump95"] - thresholds["max_wrist_jump95"]) / 4.0)
    if metrics["shoulder_width"] < thresholds["min_shoulder_width"]:
        flags.append("suspicious_pose_scale")
        penalty += (thresholds["min_shoulder_width"] - metrics["shoulder_width"]) * 0.5
    if metrics["shoulder_cv"] is None or metrics["shoulder_cv"] > thresholds["max_shoulder_cv"]:
        flags.append("shoulder_instability")
        shoulder_cv = metrics["shoulder_cv"] if metrics["shoulder_cv"] is not None else 1.0
        penalty += min(50.0, max(0.0, shoulder_cv - thresholds["max_shoulder_cv"]) * 120.0)

    return round(max(0.0, 100.0 - penalty), 2), flags


def audit_pose_file(path: Path, thresholds: dict[str, float]) -> dict[str, Any]:
    from pose_format import Pose

    with path.open("rb") as f:
        pose = Pose.read(f.read())

    data = pose.body.data
    confidence = pose.body.confidence
    pose_slice = component_slice(pose, "POSE_LANDMARKS")
    left_hand = component_slice(pose, "LEFT_HAND_LANDMARKS")
    right_hand = component_slice(pose, "RIGHT_HAND_LANDMARKS")
    hand_slices = [left_hand, right_hand]

    hand_frame_values = []
    hand_coverage_values = []
    for hand_slice in hand_slices:
        frame_fraction, coverage = _slice_visibility(confidence, hand_slice)
        hand_frame_values.append(frame_fraction)
        hand_coverage_values.append(coverage)

    pose_frame_fraction, pose_landmark_coverage = _slice_visibility(confidence, pose_slice)
    shoulder_width, shoulder_cv = _shoulder_metrics(pose, pose_slice)
    jump_candidates = [_jump95(data, confidence, pose_slice), _jump95(data, confidence, left_hand), _jump95(data, confidence, right_hand)]

    metrics = {
        "status": "ok",
        "frames": int(data.shape[0]),
        "fps": safe_float(float(pose.body.fps)),
        "hand_frame_fraction": max(hand_frame_values) if hand_frame_values else 0.0,
        "hand_landmark_coverage": max(hand_coverage_values) if hand_coverage_values else 0.0,
        "pose_frame_fraction": pose_frame_fraction,
        "pose_landmark_coverage": pose_landmark_coverage,
        "shoulder_width": shoulder_width,
        "shoulder_cv": safe_float(shoulder_cv),
        "jump95": max(jump_candidates),
        "wrist_jump95": _wrist_jump95(pose, hand_slices, pose_slice),
    }
    score, flags = _quality_score(metrics, thresholds)
    metrics["quality_score"] = score
    metrics["flags"] = flags
    return metrics


def audit_lexicon(lexicon_dir: Path, thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_index(lexicon_dir)
    records: list[dict[str, Any]] = []

    indexed_paths = set()
    for row in rows:
        rel_path = row.get("path", "")
        indexed_paths.add(rel_path)
        gloss = gloss_slug(row.get("glosses") or row.get("words") or Path(rel_path).stem)
        abs_path = lexicon_dir / rel_path
        base = {
            "gloss": gloss,
            "path": rel_path,
        }
        if not rel_path or not abs_path.is_file():
            metrics = {
                "status": "missing_pose_file",
                "frames": 0,
                "fps": None,
                "hand_frame_fraction": 0.0,
                "hand_landmark_coverage": 0.0,
                "pose_frame_fraction": 0.0,
                "pose_landmark_coverage": 0.0,
                "shoulder_width": 0.0,
                "shoulder_cv": None,
                "jump95": 0.0,
                "wrist_jump95": 0.0,
                "quality_score": 0.0,
                "flags": ["missing_pose_data"],
            }
        else:
            try:
                metrics = audit_pose_file(abs_path, thresholds)
            except Exception as exc:
                metrics = {
                    "status": "unreadable_pose_file",
                    "error": str(exc)[:180],
                    "frames": 0,
                    "fps": None,
                    "hand_frame_fraction": 0.0,
                    "hand_landmark_coverage": 0.0,
                    "pose_frame_fraction": 0.0,
                    "pose_landmark_coverage": 0.0,
                    "shoulder_width": 0.0,
                    "shoulder_cv": None,
                    "jump95": 0.0,
                    "wrist_jump95": 0.0,
                    "quality_score": 0.0,
                    "flags": ["suspicious_pose_data"],
                }
        records.append(base | metrics)

    pose_files = {str(path.relative_to(lexicon_dir)) for path in (lexicon_dir / "ase").glob("*.pose")}
    orphans = sorted(pose_files - indexed_paths)
    records.sort(key=lambda item: (item["quality_score"], item["gloss"]))

    summary = {
        "index_rows": len(rows),
        "pose_files": len(pose_files),
        "orphan_pose_files": len(orphans),
        "audited_entries": len(records),
        "mean_quality_score": round(float(np.mean([r["quality_score"] for r in records])), 2) if records else 0.0,
        "flag_counts": {},
    }
    for record in records:
        for flag in record["flags"]:
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + 1
    return records, summary


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "gloss",
        "path",
        "quality_score",
        "flags",
        "frames",
        "hand_frame_fraction",
        "hand_landmark_coverage",
        "pose_frame_fraction",
        "pose_landmark_coverage",
        "jump95",
        "wrist_jump95",
        "shoulder_width",
        "shoulder_cv",
        "status",
    ]
    out = {}
    for field in fields:
        value = record.get(field)
        out[field] = safe_float(value) if isinstance(value, float) else value
    return out


def build_quality_report(lexicon_dir: Path, limit: int, thresholds: dict[str, float]) -> dict[str, Any]:
    records, summary = audit_lexicon(lexicon_dir, thresholds)
    categories = {
        "missing_or_suspicious_pose_data": [r for r in records if r["status"] != "ok"],
        "lowest_frame_count": sorted(records, key=lambda r: (r["frames"], r["quality_score"], r["gloss"])),
        "lowest_hand_coverage": sorted(records, key=lambda r: (r["hand_frame_fraction"], r["quality_score"], r["gloss"])),
        "low_frame_count": [r for r in records if "low_frame_count" in r["flags"]],
        "unstable_hand_tracking": [r for r in records if "unstable_or_missing_hands" in r["flags"] or "unstable_wrist_tracking" in r["flags"]],
        "excessive_pose_jumpiness": [r for r in records if "excessive_pose_jumpiness" in r["flags"]],
        "incomplete_landmark_coverage": [r for r in records if "incomplete_hand_landmarks" in r["flags"] or "incomplete_pose_landmarks" in r["flags"]],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lexicon_dir": str(lexicon_dir),
        "thresholds": thresholds,
        "summary": summary,
        "weakest": [compact_record(r) for r in records[:limit]],
        "categories": {
            name: [compact_record(r) for r in values[:limit]]
            for name, values in categories.items()
        },
    }


def write_json(path: Path | None, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(compact_record(records[0]).keys()) if records else ["gloss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = compact_record(record)
            row["flags"] = ";".join(row["flags"])
            writer.writerow(row)


def read_terms(args) -> list[str]:
    terms = list(args.terms or [])
    if args.terms_file:
        terms.extend(
            line.strip()
            for line in Path(args.terms_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return terms


def resolver_from_args(args) -> GlossResolver:
    alias_file = args.aliases or str(Path(args.lexicon) / "aliases.json")
    return GlossResolver.from_paths(args.lexicon, alias_file)


def cmd_quality(args) -> int:
    lexicon_dir = Path(args.lexicon)
    thresholds = DEFAULT_THRESHOLDS.copy()
    report = build_quality_report(lexicon_dir, args.limit, thresholds)
    write_json(Path(args.output) if args.output else None, report)
    if args.csv_output:
        records, _summary = audit_lexicon(lexicon_dir, thresholds)
        write_csv(Path(args.csv_output), records)
    return 0


def cmd_weakest(args) -> int:
    records, summary = audit_lexicon(Path(args.lexicon), DEFAULT_THRESHOLDS.copy())
    write_json(None, {"summary": summary, "weakest": [compact_record(r) for r in records[: args.limit]]})
    return 0


def cmd_unresolved(args) -> int:
    resolver = resolver_from_args(args)
    results = []
    for term in read_terms(args):
        resolved_text, parts = resolver.resolve_text(term)
        unresolved = [p.target for p in parts if p.method == "unresolved"]
        results.append({
            "input": term,
            "resolved_text": resolved_text,
            "resolved": [p.__dict__ for p in parts],
            "unresolved": unresolved,
        })
    write_json(None, {"results": results})
    return 0


def cmd_aliases(args) -> int:
    lexicon = load_lexicon_glosses(args.lexicon)
    aliases = load_aliases(args.aliases or str(Path(args.lexicon) / "aliases.json"))
    resolver = GlossResolver(lexicon, aliases)
    invalid_targets = []
    redundant_aliases = []
    for source, target in aliases.items():
        if target not in lexicon:
            invalid_targets.append({"alias": source, "target": target, "suggestions": resolver.suggest_mappings(target)})
        elif gloss_slug(source) == target:
            redundant_aliases.append({"alias": source, "target": target})

    missing_aliases = []
    for term in read_terms(args):
        item = resolver.resolve_term(term)
        if item.method == "unresolved":
            missing_aliases.append({"term": term, "suggestions": resolver.suggest_mappings(term, args.limit)})

    write_json(None, {
        "alias_count": len(aliases),
        "invalid_targets": invalid_targets,
        "redundant_aliases": redundant_aliases,
        "missing_aliases": missing_aliases,
    })
    return 0


def cmd_suggest(args) -> int:
    resolver = resolver_from_args(args)
    write_json(None, {
        "suggestions": [
            {"term": term, "candidates": resolver.suggest_mappings(term, args.limit)}
            for term in read_terms(args)
        ]
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and improve the DeepSign WLASL lexicon.")
    parser.add_argument("--lexicon", default="python/lexicon_wlasl", help="Lexicon directory containing index.csv.")
    parser.add_argument("--aliases", default=None, help="Alias JSON file. Defaults to <lexicon>/aliases.json.")
    sub = parser.add_subparsers(dest="command", required=True)

    quality = sub.add_parser("quality", help="Write a compact machine-readable quality report.")
    quality.add_argument("--limit", type=int, default=25)
    quality.add_argument("--output", default=None)
    quality.add_argument("--csv-output", default=None, help="Optional full ranking CSV.")
    quality.set_defaults(func=cmd_quality)

    weakest = sub.add_parser("weakest", help="Show the weakest glosses as JSON.")
    weakest.add_argument("--limit", type=int, default=25)
    weakest.set_defaults(func=cmd_weakest)

    unresolved = sub.add_parser("unresolved", help="Show how user vocabulary resolves before fingerspelling.")
    unresolved.add_argument("terms", nargs="*")
    unresolved.add_argument("--terms-file", default=None)
    unresolved.set_defaults(func=cmd_unresolved)

    aliases = sub.add_parser("aliases", help="Validate aliases and show missing aliases for a term list.")
    aliases.add_argument("terms", nargs="*")
    aliases.add_argument("--terms-file", default=None)
    aliases.add_argument("--limit", type=int, default=5)
    aliases.set_defaults(func=cmd_aliases)

    suggest = sub.add_parser("suggest", help="Suggest candidate lexicon mappings for terms.")
    suggest.add_argument("terms", nargs="*")
    suggest.add_argument("--terms-file", default=None)
    suggest.add_argument("--limit", type=int, default=5)
    suggest.set_defaults(func=cmd_suggest)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
