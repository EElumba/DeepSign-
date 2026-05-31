"""Normalize .pose clip dimensions without changing aspect ratio.

The runtime lexicon can contain clips generated from source videos with
different frame sizes. This script rewrites those clips into one coordinate
space so pose-viewer sees consistent dimensions between signs.
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

from pose_format import Pose
from pose_format.pose_header import PoseHeaderDimensions


def iter_pose_files(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.pose"))


def image_space_slices(pose: Pose):
    """Return point slices for components whose x/y are stored in image pixels."""
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        name = component.name.upper()
        if "WORLD" not in name:
            yield slice(offset, offset + count)
        offset += count


def normalize_file(path: Path, width: int, height: int, dry_run: bool, backup: bool) -> str:
    with path.open("rb") as f:
        pose = Pose.read(f.read())

    dims = pose.header.dimensions
    old_width = int(dims.width)
    old_height = int(dims.height)

    if old_width == width and old_height == height:
        return "skipped"
    if old_width <= 0 or old_height <= 0:
        raise ValueError(f"invalid pose dimensions {old_width}x{old_height}")

    if dry_run:
        return f"would normalize {old_width}x{old_height}"

    scale = min(width / old_width, height / old_height)
    x_offset = (width - old_width * scale) / 2
    y_offset = (height - old_height * scale) / 2

    for point_slice in image_space_slices(pose):
        visible = pose.body.confidence[:, :, point_slice] > 0
        x = pose.body.data[:, :, point_slice, 0]
        y = pose.body.data[:, :, point_slice, 1]
        z = pose.body.data[:, :, point_slice, 2]
        x[visible] = x[visible] * scale + x_offset
        y[visible] = y[visible] * scale + y_offset
        z[visible] = z[visible] * scale

    pose.header.dimensions = PoseHeaderDimensions(width=width, height=height, depth=dims.depth)

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    buf = io.BytesIO()
    pose.write(buf)
    path.write_bytes(buf.getvalue())
    return f"normalized {old_width}x{old_height}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize .pose files to one width/height.")
    parser.add_argument("path", nargs="?", default="lexicon_wlasl", help="A .pose file or directory of .pose files.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files.")
    parser.add_argument("--backup", action="store_true", help="Write a .pose.bak copy before modifying each file.")
    parser.add_argument("--max-passes", type=int, default=10, help="Repeat until clean, up to this many passes.")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        parser.error(f"path does not exist: {root}")

    pose_files = list(iter_pose_files(root))
    total_counts = {"normalized": 0, "skipped": 0, "failed": 0}
    max_passes = 1 if args.dry_run else max(1, args.max_passes)

    for pass_index in range(max_passes):
        counts = {"normalized": 0, "skipped": 0, "failed": 0}
        if max_passes > 1:
            print(f"Pass {pass_index + 1}:")

        for pose_path in pose_files:
            try:
                result = normalize_file(pose_path, args.width, args.height, args.dry_run, args.backup)
                if result == "skipped":
                    counts["skipped"] += 1
                else:
                    counts["normalized"] += 1
                print(f"{pose_path}: {result}")
            except Exception as exc:
                counts["failed"] += 1
                print(f"{pose_path}: failed: {exc}")

        for key, value in counts.items():
            total_counts[key] += value

        if args.dry_run or counts["failed"] or counts["normalized"] == 0:
            break

    final_counts = counts if args.dry_run else total_counts
    action = "would normalize" if args.dry_run else "normalized"
    print(
        f"Done: {action} {final_counts['normalized']}, "
        f"skipped {final_counts['skipped']}, failed {final_counts['failed']}."
    )
    return 1 if final_counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
