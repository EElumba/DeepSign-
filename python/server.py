import io
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from threading import Lock
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
# describe different-sized and differently-positioned heads even after the
# canvas dimensions are normalized. Normalize the face component at runtime so
# the avatar head does not grow, shrink, or drift between words.
TARGET_FACE_WIDTH = float(os.environ.get("TARGET_FACE_WIDTH", "116"))
TARGET_FACE_HEIGHT = float(os.environ.get("TARGET_FACE_HEIGHT", "120"))
TARGET_FACE_SHOULDER_X = float(os.environ.get("TARGET_FACE_SHOULDER_X", "0"))
TARGET_FACE_SHOULDER_Y = float(os.environ.get("TARGET_FACE_SHOULDER_Y", "-94"))
STABILIZE_FACE_SIZE = os.environ.get("STABILIZE_FACE_SIZE", "1").lower() not in ("0", "false", "no")
TARGET_BODY_HEIGHT = float(os.environ.get("TARGET_BODY_HEIGHT", "210"))
STABILIZE_BODY_SIZE = os.environ.get("STABILIZE_BODY_SIZE", "1").lower() not in ("0", "false", "no")
AVATAR_Y_OFFSET = float(os.environ.get("AVATAR_Y_OFFSET", "-45"))
POSE_CACHE_SIZE = max(0, int(os.environ.get("POSE_CACHE_SIZE", "128")))
POSE_GENERATION_TIMEOUT = float(os.environ.get("POSE_GENERATION_TIMEOUT", "30"))

_pose_cache: OrderedDict[str, bytes] = OrderedDict()
_pose_cache_lock = Lock()

# --- Sign recognition lexicon (loaded at startup) ---
_lexicon_features: dict = {}   # gloss -> np.ndarray (126,) mean feature vector
_lexicon_features_lock = Lock()


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


class RecognizeRequest(BaseModel):
    frames: list  # [{left_hand: [[x,y,z]…]|null, right_hand: …}]


def _cache_get(text: str) -> Optional[bytes]:
    if POSE_CACHE_SIZE <= 0:
        return None
    with _pose_cache_lock:
        cached = _pose_cache.get(text)
        if cached is not None:
            _pose_cache.move_to_end(text)
        return cached


def _cache_set(text: str, pose_bytes: bytes) -> None:
    if POSE_CACHE_SIZE <= 0:
        return
    with _pose_cache_lock:
        _pose_cache[text] = pose_bytes
        _pose_cache.move_to_end(text)
        while len(_pose_cache) > POSE_CACHE_SIZE:
            _pose_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Sign recognition helpers
# ---------------------------------------------------------------------------

def _normalize_hand(lm: Optional[np.ndarray]) -> np.ndarray:
    """Center a hand at its wrist and normalize by max finger distance.

    Both the browser MediaPipe [0-1] space and the lexicon pixel space are
    reduced to a unit-scale, wrist-origin shape descriptor, making them
    comparable regardless of absolute coordinate system.
    """
    zeros = np.zeros((21, 3), dtype=np.float32)
    if lm is None or lm.shape != (21, 3):
        return zeros
    lm = lm.astype(np.float32)
    lm -= lm[0]  # center at wrist (index 0)
    scale = np.linalg.norm(lm, axis=1).max()
    if scale > 1e-6:
        lm /= scale
    return lm


def _frame_feature(lh: Optional[np.ndarray], rh: Optional[np.ndarray]) -> np.ndarray:
    """126-dim feature vector: normalized left hand (63) + right hand (63)."""
    return np.concatenate([_normalize_hand(lh).flatten(), _normalize_hand(rh).flatten()])


def _pose_to_mean_feature(pose) -> Optional[np.ndarray]:
    """Extract a single mean (126,) feature vector from a pose_format Pose."""
    lh_sl = _component_slice(pose, "LEFT_HAND_LANDMARKS")
    rh_sl = _component_slice(pose, "RIGHT_HAND_LANDMARKS")
    if lh_sl is None and rh_sl is None:
        return None

    feats = []
    for f in range(pose.body.data.shape[0]):
        lh_data = rh_data = None
        if lh_sl is not None and pose.body.confidence[f, 0, lh_sl].mean() > 0.1:
            lh_data = pose.body.data[f, 0, lh_sl, :3]
        if rh_sl is not None and pose.body.confidence[f, 0, rh_sl].mean() > 0.1:
            rh_data = pose.body.data[f, 0, rh_sl, :3]
        feats.append(_frame_feature(lh_data, rh_data))

    if not feats:
        return None
    mean = np.stack(feats).mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 1e-6 else None


def _load_lexicon_features() -> None:
    """Read every .pose file in the lexicon and cache its mean feature vector."""
    try:
        from pose_format import Pose as _Pose
    except ImportError:
        print("[Recognize] pose_format not installed; recognition disabled.")
        return

    index_path = os.path.join(LEXICON_DIR, "index.csv")
    if not os.path.isfile(index_path):
        print(f"[Recognize] No index.csv found at {LEXICON_DIR}; recognition disabled.")
        return

    new_features: dict = {}
    loaded = failed = 0

    with open(index_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        gloss = (row.get("glosses") or row.get("words") or "").strip().lower()
        abs_path = os.path.join(LEXICON_DIR, row.get("path", ""))
        if not gloss or not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "rb") as pf:
                pose = _Pose.read(pf.read())
            feat = _pose_to_mean_feature(pose)
            if feat is not None:
                new_features[gloss] = feat
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[Recognize] Skipping {abs_path}: {e}")
            failed += 1

    with _lexicon_features_lock:
        _lexicon_features.update(new_features)

    print(f"[Recognize] {loaded} sign feature(s) loaded, {failed} skipped.")


def generate_pose(text: str) -> bytes:
    """Run the `text_to_gloss_to_pose` CLI and return the raw .pose bytes.

    Uses a unique temp file per call so concurrent requests don't collide.
    """
    cached = _cache_get(text)
    if cached is not None:
        return cached

    fd, out_path = tempfile.mkstemp(suffix=".pose")
    os.close(fd)
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
            timeout=POSE_GENERATION_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"text_to_gloss_to_pose failed: {result.stderr}")
        with open(out_path, "rb") as f:
            pose_bytes = _postprocess_pose(f.read())
        _cache_set(text, pose_bytes)
        return pose_bytes
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _postprocess_pose(pose_bytes: bytes) -> bytes:
    """Apply runtime cleanup passes to generated pose bytes."""
    try:
        from pose_format import Pose

        pose = Pose.read(pose_bytes)
        if STABILIZE_BODY_SIZE:
            _stabilize_body_size(pose)
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


def _component_points(pose, component_name: str):
    for component in pose.header.components:
        if component.name == component_name:
            return list(component.points)
    return None


def _image_space_slices(pose):
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        if "WORLD" not in component.name.upper():
            yield slice(offset, offset + count)
        offset += count


def _body_adjustment_slices(pose):
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        name = component.name.upper()
        if "WORLD" not in name and name != "FACE_LANDMARKS":
            yield slice(offset, offset + count)
        offset += count


def _shift_avatar_y(pose, offset: float) -> None:
    """Shift image-space landmarks vertically inside the pose canvas."""
    for point_slice in _image_space_slices(pose):
        visible = pose.body.confidence[:, :, point_slice] > 0
        y = pose.body.data[:, :, point_slice, 1]
        y[visible] = y[visible] + offset


def _stabilize_body_size(pose) -> None:
    """Keep shoulder-to-hip body length stable without changing shoulder width."""
    if TARGET_BODY_HEIGHT <= 0:
        return

    pose_slice = _component_slice(pose, "POSE_LANDMARKS")
    pose_points = _component_points(pose, "POSE_LANDMARKS") or []
    if pose_slice is None:
        return
    try:
        left_shoulder_index = pose_points.index("LEFT_SHOULDER")
        right_shoulder_index = pose_points.index("RIGHT_SHOULDER")
        left_hip_index = pose_points.index("LEFT_HIP")
        right_hip_index = pose_points.index("RIGHT_HIP")
    except ValueError:
        return

    pose_data = pose.body.data[:, :, pose_slice, :]
    pose_confidence = pose.body.confidence[:, :, pose_slice]
    adjustment_slices = list(_body_adjustment_slices(pose))
    required = (
        left_shoulder_index,
        right_shoulder_index,
        left_hip_index,
        right_hip_index,
    )
    valid = (pose_confidence[:, :, required] > 0).all(axis=2)
    if not valid.any():
        return

    left_shoulder = pose_data[:, :, left_shoulder_index, :2]
    right_shoulder = pose_data[:, :, right_shoulder_index, :2]
    left_hip = pose_data[:, :, left_hip_index, :2]
    right_hip = pose_data[:, :, right_hip_index, :2]
    valid &= (
        np.isfinite(left_shoulder).all(axis=2)
        & np.isfinite(right_shoulder).all(axis=2)
        & np.isfinite(left_hip).all(axis=2)
        & np.isfinite(right_hip).all(axis=2)
    )

    shoulder_y = (left_shoulder[:, :, 1] + right_shoulder[:, :, 1]) / 2
    hip_y = (left_hip[:, :, 1] + right_hip[:, :, 1]) / 2
    body_height = hip_y - shoulder_y
    valid &= body_height > 1
    if not valid.any():
        return

    scale_y = np.ones_like(body_height, dtype=np.float32)
    scale_y[valid] = np.clip(TARGET_BODY_HEIGHT / body_height[valid], 0.65, 1.35)
    shoulder_y = shoulder_y[:, :, None]
    scale_y = scale_y[:, :, None]
    valid = valid[:, :, None]

    for point_slice in adjustment_slices:
        visible = pose.body.confidence[:, :, point_slice] > 0
        mask = visible & valid
        if not mask.any():
            continue
        points = pose.body.data[:, :, point_slice]
        points[:, :, :, 1] = np.where(
            mask,
            (points[:, :, :, 1] - shoulder_y) * scale_y + shoulder_y,
            points[:, :, :, 1],
        )
        points[:, :, :, 2] = np.where(
            mask,
            points[:, :, :, 2] * scale_y,
            points[:, :, :, 2],
        )


def _stabilize_face_size(pose) -> None:
    """Keep face landmark shape and shoulder-relative placement stable."""
    if TARGET_FACE_WIDTH <= 0 and TARGET_FACE_HEIGHT <= 0:
        return

    face_slice = _component_slice(pose, "FACE_LANDMARKS")
    if face_slice is None:
        return
    pose_slice = _component_slice(pose, "POSE_LANDMARKS")
    pose_points = _component_points(pose, "POSE_LANDMARKS") or []
    try:
        left_shoulder_index = pose_points.index("LEFT_SHOULDER")
        right_shoulder_index = pose_points.index("RIGHT_SHOULDER")
    except ValueError:
        left_shoulder_index = right_shoulder_index = None

    data = pose.body.data[:, :, face_slice, :]
    confidence = pose.body.confidence[:, :, face_slice]
    visible = confidence > 0
    pose_data = pose.body.data[:, :, pose_slice, :] if pose_slice is not None else None
    pose_confidence = pose.body.confidence[:, :, pose_slice] if pose_slice is not None else None

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
            face_width = float(max_xy[0] - min_xy[0])
            face_height = float(max_xy[1] - min_xy[1])
            if face_width <= 1 or face_height <= 1:
                continue

            scale_x = TARGET_FACE_WIDTH / face_width if TARGET_FACE_WIDTH > 0 else 1.0
            scale_y = TARGET_FACE_HEIGHT / face_height if TARGET_FACE_HEIGHT > 0 else 1.0
            if not np.isfinite(scale_x) or not np.isfinite(scale_y):
                continue
            # Keep outlier detections from causing dramatic warps.
            scale_x = float(np.clip(scale_x, 0.65, 1.45))
            scale_y = float(np.clip(scale_y, 0.65, 1.45))

            center = (min_xy + max_xy) / 2
            points[visible_indices, 0] = (points[visible_indices, 0] - center[0]) * scale_x + center[0]
            points[visible_indices, 1] = (points[visible_indices, 1] - center[1]) * scale_y + center[1]
            points[visible_indices, 2] = points[visible_indices, 2] * ((scale_x + scale_y) / 2)

            if (
                pose_data is None
                or pose_confidence is None
                or left_shoulder_index is None
                or right_shoulder_index is None
                or pose_confidence[frame_index, person_index, left_shoulder_index] <= 0
                or pose_confidence[frame_index, person_index, right_shoulder_index] <= 0
            ):
                continue

            left_shoulder = pose_data[frame_index, person_index, left_shoulder_index, :2]
            right_shoulder = pose_data[frame_index, person_index, right_shoulder_index, :2]
            if not np.isfinite(left_shoulder).all() or not np.isfinite(right_shoulder).all():
                continue

            shoulder_center = (left_shoulder + right_shoulder) / 2
            target_center = shoulder_center + np.array([TARGET_FACE_SHOULDER_X, TARGET_FACE_SHOULDER_Y])
            adjusted_xy = points[visible_indices, :2]
            adjusted_xy = adjusted_xy[np.isfinite(adjusted_xy).all(axis=1)]
            if len(adjusted_xy) < 20:
                continue
            adjusted_min_xy = np.percentile(adjusted_xy, 2, axis=0)
            adjusted_max_xy = np.percentile(adjusted_xy, 98, axis=0)
            adjusted_center = (adjusted_min_xy + adjusted_max_xy) / 2
            points[visible_indices, :2] = points[visible_indices, :2] + (target_center - adjusted_center)


@app.on_event("startup")
async def startup():
    print(f"[Startup] ASL Pose Server listening on port {PORT}")
    print(f"[Startup] text_to_gloss_to_pose CLI: {CLI}")
    print(f"[Startup] Lexicon dir: {LEXICON_DIR}")
    try:
        generate_pose("hello")
        print("[Warmup] Pose model ready")
    except Exception as e:
        print(f"[Warmup] Warning: {e}")
    # Load lexicon features for sign recognition (runs in the startup thread).
    _load_lexicon_features()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recognize")
def recognize_sign(req: RecognizeRequest):
    with _lexicon_features_lock:
        n_signs = len(_lexicon_features)
    if n_signs == 0:
        raise HTTPException(status_code=503, detail="Recognition lexicon not loaded")
    if not req.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    feats = []
    for frame in req.frames:
        if not isinstance(frame, dict):
            continue
        lh = frame.get("left_hand")
        rh = frame.get("right_hand")
        lh_arr = np.array(lh, dtype=np.float32) if (lh and len(lh) == 21) else None
        rh_arr = np.array(rh, dtype=np.float32) if (rh and len(rh) == 21) else None
        feats.append(_frame_feature(lh_arr, rh_arr))

    if not feats:
        return {"gloss": "", "confidence": 0.0}

    query_mean = np.stack(feats).mean(axis=0)
    norm = np.linalg.norm(query_mean)
    if norm < 1e-6:
        return {"gloss": "", "confidence": 0.0}
    query_mean = (query_mean / norm).astype(np.float32)

    best_gloss, best_score = "", -1.0
    with _lexicon_features_lock:
        for gloss, ref in _lexicon_features.items():
            score = float(np.dot(query_mean, ref))
            if score > best_score:
                best_score, best_gloss = score, gloss

    return {"gloss": best_gloss, "confidence": round(best_score, 4)}


@app.post("/pose")
def pose(req: PoseRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        pose_bytes = generate_pose(text)
    except Exception as e:
        print(f"[Pose] Error generating pose: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=pose_bytes, media_type="application/octet-stream")
