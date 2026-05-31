import io
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

PORT = int(os.environ.get("PORT", "8000"))

# Playback speed multiplier for the signing avatar. pose-viewer animates at the
# pose's stored fps, so raising fps makes the avatar sign faster without
# dropping any motion. 1.0 = native speed; 1.6 ≈ 60% faster. Tune via env.
SIGN_SPEED = float(os.environ.get("SIGN_SPEED", "1.6"))

# WLASL clips come from different videos/signers, so their face landmarks can
# describe different-sized heads even after the canvas dimensions are normalized.
# Normalize the face component at runtime so the avatar head does not grow or
# shrink between words.
TARGET_FACE_HEIGHT = float(os.environ.get("TARGET_FACE_HEIGHT", "120"))
STABILIZE_FACE_SIZE = os.environ.get("STABILIZE_FACE_SIZE", "1").lower() not in ("0", "false", "no")
AVATAR_Y_OFFSET = float(os.environ.get("AVATAR_Y_OFFSET", "-45"))


def _fingerspelling_lexicon_dir() -> str:
    import spoken_to_signed
    return os.path.join(
        os.path.dirname(spoken_to_signed.__file__),
        "assets",
        "fingerspelling_lexicon",
    )


def _resolve_lexicon_dir() -> str:
    """Pick the --lexicon directory passed to `text_to_gloss_to_pose`.

    Precedence:
      1. LEXICON_DIR env var (explicit override).
      2. The WLASL word-level lexicon built by build_lexicon.py, if present
         (real two-handed signs; out-of-vocabulary words still fall back to
         fingerspelling automatically inside the CLI).
      3. The fingerspelling lexicon bundled with `spoken_to_signed`
         (every word signed letter-by-letter).

    Fingerspelling is always available as a fallback regardless of which
    directory is selected, because the CLI wraps the chosen lexicon with a
    FingerspellingPoseLookup backup.
    """
    override = os.environ.get("LEXICON_DIR")
    if override:
        return override

    wlasl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon_wlasl")
    if _lexicon_has_usable_entries(wlasl_dir):
        return wlasl_dir

    return _fingerspelling_lexicon_dir()


def _lexicon_has_usable_entries(directory: str) -> bool:
    index_path = os.path.join(directory, "index.csv")
    if not os.path.isfile(index_path):
        return False

    try:
        with open(index_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pose_path = row.get("path")
                if pose_path and os.path.isfile(os.path.join(directory, pose_path)):
                    return True
    except Exception as e:
        print(f"[Startup] Ignoring unusable lexicon at {directory}: {e}")
    return False


LEXICON_DIR = _resolve_lexicon_dir()


def _resolve_cli() -> str:
    """Find the `text_to_gloss_to_pose` CLI. Prefer the executable installed
    alongside the running Python interpreter (i.e. the venv's bin/Scripts dir),
    so it works even when the venv isn't activated. Fall back to PATH lookup.
    """
    bin_dir = os.path.dirname(sys.executable)
    for name in ("text_to_gloss_to_pose", "text_to_gloss_to_pose.exe"):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("text_to_gloss_to_pose") or "text_to_gloss_to_pose"


CLI = _resolve_cli()

app = FastAPI(title="ASL Pose Server")

# The Node server (port 3000) calls this server via node-fetch, so allow all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PoseRequest(BaseModel):
    text: str


def generate_pose(text: str) -> bytes:
    """Run the `text_to_gloss_to_pose` CLI and return the raw .pose bytes.

    Uses a unique temp file per call so concurrent requests don't collide.
    """
    out_path = tempfile.mktemp(suffix=".pose")
    try:
        result = subprocess.run(
            [
                CLI,
                "--text", text,
                "--glosser", "simple",
                "--spoken-language", "en",
                "--signed-language", "ase",
                "--lexicon", LEXICON_DIR,
                "--pose", out_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"text_to_gloss_to_pose failed: {result.stderr}")
        with open(out_path, "rb") as f:
            return _postprocess_pose(f.read())
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _postprocess_pose(pose_bytes: bytes) -> bytes:
    """Apply runtime cleanup passes to generated pose bytes."""
    try:
        from pose_format import Pose

        pose = Pose.read(pose_bytes)
        if STABILIZE_FACE_SIZE:
            _stabilize_face_size(pose)
        if AVATAR_Y_OFFSET:
            _shift_avatar_y(pose, AVATAR_Y_OFFSET)
        if SIGN_SPEED != 1.0:
            pose.body.fps = pose.body.fps * SIGN_SPEED

        buf = io.BytesIO()
        pose.write(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[Pose] Postprocess skipped: {e}")
        return pose_bytes


def _component_slice(pose, component_name: str) -> Optional[slice]:
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        if component.name == component_name:
            return slice(offset, offset + count)
        offset += count
    return None


def _image_space_slices(pose):
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        if "WORLD" not in component.name.upper():
            yield slice(offset, offset + count)
        offset += count


def _shift_avatar_y(pose, offset: float) -> None:
    """Shift image-space landmarks vertically inside the pose canvas."""
    for point_slice in _image_space_slices(pose):
        visible = pose.body.confidence[:, :, point_slice] > 0
        y = pose.body.data[:, :, point_slice, 1]
        y[visible] = y[visible] + offset


def _stabilize_face_size(pose) -> None:
    """Keep face landmark height stable while preserving per-frame head motion."""
    if TARGET_FACE_HEIGHT <= 0:
        return

    face_slice = _component_slice(pose, "FACE_LANDMARKS")
    if face_slice is None:
        return

    data = pose.body.data[:, :, face_slice, :]
    confidence = pose.body.confidence[:, :, face_slice]
    visible = confidence > 0

    for frame_index in range(data.shape[0]):
        for person_index in range(data.shape[1]):
            frame_visible = visible[frame_index, person_index]
            if frame_visible.sum() < 20:
                continue

            points = data[frame_index, person_index]
            frame_confidence = confidence[frame_index, person_index]
            visible_indices = np.flatnonzero(frame_visible)
            xy = points[visible_indices, :2]
            valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 0)
            invalid_indices = visible_indices[~valid]
            if len(invalid_indices):
                frame_confidence[invalid_indices] = 0
            visible_indices = visible_indices[valid]
            xy = xy[valid]
            if len(xy) < 20:
                continue

            min_xy = np.percentile(xy, 2, axis=0)
            max_xy = np.percentile(xy, 98, axis=0)
            face_height = float(max_xy[1] - min_xy[1])
            if face_height <= 1:
                continue

            scale = TARGET_FACE_HEIGHT / face_height
            if not np.isfinite(scale):
                continue
            # Keep outlier detections from causing dramatic warps.
            scale = float(np.clip(scale, 0.65, 1.45))

            center = (min_xy + max_xy) / 2
            points[visible_indices, :2] = (points[visible_indices, :2] - center) * scale + center
            points[visible_indices, 2] = points[visible_indices, 2] * scale


@app.on_event("startup")
async def startup():
    print(f"[Startup] ASL Pose Server listening on port {PORT}")
    print(f"[Startup] text_to_gloss_to_pose CLI: {CLI}")
    print(f"[Startup] Lexicon dir: {LEXICON_DIR}")
    # Warm up the pipeline (loads lexicon/pose assets) so the first real request is fast.
    try:
        generate_pose("hello")
        print("[Warmup] Pose model ready")
    except Exception as e:
        print(f"[Warmup] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/pose")
async def pose(req: PoseRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        pose_bytes = generate_pose(text)
    except Exception as e:
        print(f"[Pose] Error generating pose: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=pose_bytes, media_type="application/octet-stream")
