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
RECOGNIZE_SEQUENCE_LENGTH = max(8, int(os.environ.get("RECOGNIZE_SEQUENCE_LENGTH", "24")))
RECOGNIZE_TOP_K = max(1, int(os.environ.get("RECOGNIZE_TOP_K", "48")))
RECOGNIZE_TEMPORAL_WEIGHT = float(os.environ.get("RECOGNIZE_TEMPORAL_WEIGHT", "0.65"))
RECOGNIZE_TEMPORAL_WEIGHT = min(1.0, max(0.0, RECOGNIZE_TEMPORAL_WEIGHT))
RECOGNIZE_DTW_BAND = max(1, int(os.environ.get("RECOGNIZE_DTW_BAND", "6")))
HAND_CONFIDENCE_MIN = float(os.environ.get("HAND_CONFIDENCE_MIN", "0.1"))
TRAJECTORY_FEATURE_WEIGHT = float(os.environ.get("TRAJECTORY_FEATURE_WEIGHT", "2.5"))
VISIBILITY_FEATURE_WEIGHT = float(os.environ.get("VISIBILITY_FEATURE_WEIGHT", "0.25"))

# Replace MediaPipe Holistic's dense 478-point face mesh (the "Google" tessellation,
# ~2500 connections) with a clean ~14-point cartoon face: outline, eyes, nose, and
# mouth. Far more legible on the avatar — especially on the additive glasses display.
SIMPLE_FACE = os.environ.get("SIMPLE_FACE", "1").lower() not in ("0", "false", "no")

# Ordered (label, MediaPipe FaceMesh index) for the simplified face. The limb
# indices below are positions into THIS list, so order matters.
SIMPLE_FACE_POINTS = [
    ("face_top", "10"), ("chin", "152"), ("cheek_r", "234"), ("cheek_l", "454"),
    ("eye_r_outer", "33"), ("eye_r_inner", "133"),
    ("eye_l_inner", "362"), ("eye_l_outer", "263"),
    ("nose_top", "168"), ("nose_tip", "1"),
    ("mouth_r", "61"), ("mouth_l", "291"), ("mouth_top", "13"), ("mouth_bottom", "14"),
]
SIMPLE_FACE_LIMBS = [
    (0, 2), (2, 1), (1, 3), (3, 0),          # face outline
    (4, 5), (6, 7),                          # eyes
    (8, 9),                                  # nose bridge → tip
    (10, 12), (12, 11), (11, 13), (13, 10),  # mouth
]
SIMPLE_FACE_COLOR = (255, 255, 255)

_pose_cache: OrderedDict[str, bytes] = OrderedDict()
_pose_cache_lock = Lock()

# --- Sign recognition lexicon (loaded at startup) ---
_lexicon_features: dict = {}   # gloss -> {"mean": np.ndarray, "sequence": np.ndarray}
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
    if lm is None or lm.shape != (21, 3) or not np.isfinite(lm).all():
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


def _valid_hand(lm: Optional[np.ndarray]) -> bool:
    return lm is not None and lm.shape == (21, 3) and np.isfinite(lm).all()


def _hand_scale(lm: np.ndarray) -> float:
    centered = lm.astype(np.float32) - lm[0]
    return float(np.linalg.norm(centered, axis=1).max())


def _normalize_rows(sequence: np.ndarray) -> np.ndarray:
    sequence = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(sequence, axis=1, keepdims=True)
    normalized = np.divide(
        sequence,
        norms,
        out=np.zeros_like(sequence, dtype=np.float32),
        where=norms > 1e-6,
    ).astype(np.float32)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _resample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a variable-length sign to a fixed temporal template."""
    if sequence.shape[0] == length:
        return sequence.astype(np.float32)
    if sequence.shape[0] == 1:
        return np.repeat(sequence, length, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    out = np.empty((length, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, sequence[:, dim])
    return out


def _relative_wrist_path(wrists: np.ndarray, visible: np.ndarray, scale: float) -> np.ndarray:
    path = np.zeros((len(wrists), 2), dtype=np.float32)
    if not visible.any():
        return path
    origin = wrists[np.flatnonzero(visible)[0]]
    path[visible] = (wrists[visible] - origin) / scale
    return np.clip(path, -4.0, 4.0)


def _hand_frames_to_features(hand_frames: list) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return a mean hand-shape vector plus a fixed-length temporal template.

    The mean vector keeps the current fast shortlist behavior. The sequence
    template preserves order and wrist motion so signs with similar average
    handshape can still separate during reranking.
    """
    shape_feats = []
    left_wrists = []
    right_wrists = []
    visibility = []
    scales = []

    for lh, rh in hand_frames:
        lh_valid = _valid_hand(lh)
        rh_valid = _valid_hand(rh)
        if not lh_valid and not rh_valid:
            continue

        shape_feats.append(_frame_feature(lh if lh_valid else None, rh if rh_valid else None))
        left_wrists.append(lh[0, :2] if lh_valid else [np.nan, np.nan])
        right_wrists.append(rh[0, :2] if rh_valid else [np.nan, np.nan])
        visibility.append([1.0 if lh_valid else 0.0, 1.0 if rh_valid else 0.0])
        if lh_valid:
            scales.append(_hand_scale(lh))
        if rh_valid:
            scales.append(_hand_scale(rh))

    if not shape_feats:
        return None, None

    shape = np.stack(shape_feats).astype(np.float32)
    mean = shape.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm < 1e-6:
        return None, None
    mean = (mean / mean_norm).astype(np.float32)

    scale = float(np.median([s for s in scales if s > 1e-6])) if scales else 1.0
    if scale <= 1e-6 or not np.isfinite(scale):
        scale = 1.0

    left_wrists = np.array(left_wrists, dtype=np.float32)
    right_wrists = np.array(right_wrists, dtype=np.float32)
    visibility = np.array(visibility, dtype=np.float32)
    left_visible = visibility[:, 0] > 0
    right_visible = visibility[:, 1] > 0
    left_path = _relative_wrist_path(left_wrists, left_visible, scale)
    right_path = _relative_wrist_path(right_wrists, right_visible, scale)

    sequence = np.concatenate(
        [
            shape,
            np.concatenate([left_path, right_path], axis=1) * TRAJECTORY_FEATURE_WEIGHT,
            visibility * VISIBILITY_FEATURE_WEIGHT,
        ],
        axis=1,
    ).astype(np.float32)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    sequence = _resample_sequence(sequence, RECOGNIZE_SEQUENCE_LENGTH)
    sequence = _normalize_rows(sequence)
    return mean, sequence


def _pose_to_reference_features(pose) -> Optional[dict]:
    """Extract coarse and temporal recognition features from a pose_format Pose."""
    lh_sl = _component_slice(pose, "LEFT_HAND_LANDMARKS")
    rh_sl = _component_slice(pose, "RIGHT_HAND_LANDMARKS")
    if lh_sl is None and rh_sl is None:
        return None

    hand_frames = []
    for f in range(pose.body.data.shape[0]):
        lh_data = rh_data = None
        if lh_sl is not None and pose.body.confidence[f, 0, lh_sl].mean() > HAND_CONFIDENCE_MIN:
            lh_data = pose.body.data[f, 0, lh_sl, :3]
        if rh_sl is not None and pose.body.confidence[f, 0, rh_sl].mean() > HAND_CONFIDENCE_MIN:
            rh_data = pose.body.data[f, 0, rh_sl, :3]
        hand_frames.append((lh_data, rh_data))

    mean, sequence = _hand_frames_to_features(hand_frames)
    if mean is None or sequence is None:
        return None
    return {"mean": mean, "sequence": sequence}


def _temporal_similarity(query: np.ndarray, reference: np.ndarray) -> float:
    """Banded DTW over normalized frame descriptors, returned as average cosine."""
    if query is None or reference is None or not len(query) or not len(reference):
        return -1.0

    query = np.clip(_normalize_rows(query), -1.0, 1.0)
    reference = np.clip(_normalize_rows(reference), -1.0, 1.0)
    n, m = query.shape[0], reference.shape[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        similarity = np.clip(query @ reference.T, -1.0, 1.0)
    similarity = np.nan_to_num(similarity, nan=-1.0, posinf=1.0, neginf=-1.0)
    cost = 1.0 - similarity
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    steps = np.zeros((n + 1, m + 1), dtype=np.int16)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - RECOGNIZE_DTW_BAND)
        j_end = min(m, i + RECOGNIZE_DTW_BAND) + 1
        for j in range(j_start, j_end):
            candidates = (
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1]),
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            prev_cost, prev_steps = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost[i - 1, j - 1] + prev_cost
            steps[i, j] = prev_steps + 1

    if not np.isfinite(dp[n, m]) or steps[n, m] <= 0:
        return float(np.mean(np.sum(query * reference, axis=1)))
    avg_cost = float(dp[n, m] / steps[n, m])
    return float(np.clip(1.0 - avg_cost, -1.0, 1.0))


def _load_lexicon_features() -> None:
    """Read every .pose file in the lexicon and cache hybrid recognizer features."""
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
            features = _pose_to_reference_features(pose)
            if features is not None:
                new_features[gloss] = features
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[Recognize] Skipping {abs_path}: {e}")
            failed += 1

    with _lexicon_features_lock:
        _lexicon_features.update(new_features)

    print(
        f"[Recognize] {loaded} temporal sign feature(s) loaded, {failed} skipped "
        f"(top_k={RECOGNIZE_TOP_K}, sequence={RECOGNIZE_SEQUENCE_LENGTH}, "
        f"temporal_weight={RECOGNIZE_TEMPORAL_WEIGHT:.2f})."
    )


def generate_pose(text: str, retries: int = 2) -> bytes:
    """Run the `text_to_gloss_to_pose` CLI and return the raw .pose bytes.

    Uses a unique temp file per call so concurrent requests don't collide.

    `spoken-to-signed` reads pose files across a thread pool, and the
    pose_format binary reader occasionally races on long, fingerspelling-heavy
    utterances ("buffer is too small for requested array"). The failure is
    transient — a retry reliably succeeds — so we retry a couple of times
    before giving up.
    """
    cached = _cache_get(text)
    if cached is not None:
        return cached

    last_err = None
    for attempt in range(retries + 1):
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
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"[Pose] attempt {attempt + 1} failed ({str(e)[:80]}); retrying")
                continue
            raise
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
    raise last_err


def _postprocess_pose(pose_bytes: bytes) -> bytes:
    """Apply runtime cleanup passes to generated pose bytes."""
    try:
        from pose_format import Pose

        pose = Pose.read(pose_bytes)
        if STABILIZE_BODY_SIZE:
            _stabilize_body_size(pose)
        if STABILIZE_FACE_SIZE:
            _stabilize_face_size(pose)
        # Simplify after stabilization (which needs the dense mesh) but before the
        # Y-shift, which operates on whatever components the pose currently has.
        if SIMPLE_FACE:
            pose = _simplify_face(pose)
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


def _simplify_face(pose):
    """Swap the dense FACE_LANDMARKS mesh for a clean ~14-point cartoon face.

    Returns a new Pose with the simplified face, or the original pose unchanged
    if the face component is missing or any expected landmark is absent.
    """
    from pose_format.pose_header import PoseHeaderComponent

    face = next((c for c in pose.header.components if c.name == "FACE_LANDMARKS"), None)
    if face is None:
        return pose

    available = set(face.points)
    selected = [mp_index for _, mp_index in SIMPLE_FACE_POINTS]
    if not all(idx in available for idx in selected):
        # Unexpected face point layout — leave the pose as-is rather than
        # building a face with misaligned limbs.
        return pose

    component_names = [c.name for c in pose.header.components]
    reduced = pose.get_components(component_names, {"FACE_LANDMARKS": selected})

    for i, component in enumerate(reduced.header.components):
        if component.name == "FACE_LANDMARKS":
            reduced.header.components[i] = PoseHeaderComponent(
                "FACE_LANDMARKS",
                component.points,
                list(SIMPLE_FACE_LIMBS),
                [SIMPLE_FACE_COLOR],
                component.format,
            )
            break
    return reduced


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

    hand_frames = []
    for frame in req.frames:
        if not isinstance(frame, dict):
            continue
        lh = frame.get("left_hand")
        rh = frame.get("right_hand")
        lh_arr = np.array(lh, dtype=np.float32) if (lh and len(lh) == 21) else None
        rh_arr = np.array(rh, dtype=np.float32) if (rh and len(rh) == 21) else None
        hand_frames.append((lh_arr, rh_arr))

    query_mean, query_sequence = _hand_frames_to_features(hand_frames)
    if query_mean is None or query_sequence is None:
        return {"gloss": "", "confidence": 0.0}

    candidates = []
    with _lexicon_features_lock:
        for gloss, ref in _lexicon_features.items():
            mean_score = float(np.dot(query_mean, ref["mean"]))
            candidates.append((mean_score, gloss, ref))

    if not candidates:
        return {"gloss": "", "confidence": 0.0}

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = {
        "gloss": "",
        "score": -1.0,
        "mean_score": -1.0,
        "temporal_score": -1.0,
    }

    for mean_score, gloss, ref in candidates[:RECOGNIZE_TOP_K]:
        temporal_score = _temporal_similarity(query_sequence, ref["sequence"])
        combined_score = (
            (1.0 - RECOGNIZE_TEMPORAL_WEIGHT) * mean_score
            + RECOGNIZE_TEMPORAL_WEIGHT * temporal_score
        )
        if combined_score > best["score"]:
            best = {
                "gloss": gloss,
                "score": combined_score,
                "mean_score": mean_score,
                "temporal_score": temporal_score,
            }

    return {
        "gloss": best["gloss"],
        "confidence": round(float(best["score"]), 4),
        "mean_confidence": round(float(best["mean_score"]), 4),
        "temporal_confidence": round(float(best["temporal_score"]), 4),
        "candidates_considered": min(RECOGNIZE_TOP_K, len(candidates)),
    }


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
