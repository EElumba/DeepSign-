import io
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from threading import Lock, Thread
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from lexicon_resolution import GlossResolver

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
POSE_LOOKUP_CACHE_SIZE = max(1, int(os.environ.get("POSE_LOOKUP_CACHE_SIZE", "256")))
POSE_BACKEND = os.environ.get("POSE_BACKEND", "library").strip().lower()
POSE_LIBRARY_LOGS = os.environ.get("POSE_LIBRARY_LOGS", "0").lower() in ("1", "true", "yes")
POSE_GENERATION_TIMEOUT = float(os.environ.get("POSE_GENERATION_TIMEOUT", "30"))
RECOGNIZE_SEQUENCE_LENGTH = max(8, int(os.environ.get("RECOGNIZE_SEQUENCE_LENGTH", "24")))
RECOGNIZE_TOP_K = max(1, int(os.environ.get("RECOGNIZE_TOP_K", "48")))
RECOGNIZE_TEMPORAL_WEIGHT = float(os.environ.get("RECOGNIZE_TEMPORAL_WEIGHT", "0.65"))
RECOGNIZE_TEMPORAL_WEIGHT = min(1.0, max(0.0, RECOGNIZE_TEMPORAL_WEIGHT))
RECOGNIZE_DTW_BAND = max(1, int(os.environ.get("RECOGNIZE_DTW_BAND", "6")))
RECOGNIZE_CONFIDENCE_MIN = float(os.environ.get("RECOGNIZE_CONFIDENCE_MIN", "0.70"))
RECOGNIZE_MARGIN_MIN = float(os.environ.get("RECOGNIZE_MARGIN_MIN", "0.04"))
RECOGNIZE_HIGH_CONFIDENCE_MIN = float(os.environ.get("RECOGNIZE_HIGH_CONFIDENCE_MIN", "0.92"))
RECOGNIZE_HIGH_CONFIDENCE_MARGIN_MIN = float(os.environ.get("RECOGNIZE_HIGH_CONFIDENCE_MARGIN_MIN", "0.02"))
RECOGNIZE_MIN_VISIBLE_FRAMES = max(1, int(os.environ.get("RECOGNIZE_MIN_VISIBLE_FRAMES", "8")))
RECOGNIZE_MIN_VISIBLE_RATIO = float(os.environ.get("RECOGNIZE_MIN_VISIBLE_RATIO", "0.45"))
RECOGNIZE_MIN_VISIBLE_RATIO = min(1.0, max(0.0, RECOGNIZE_MIN_VISIBLE_RATIO))
RECOGNIZE_MIN_MOTION_PATH = float(os.environ.get("RECOGNIZE_MIN_MOTION_PATH", "0.08"))
RECOGNIZE_MIN_HANDSHAPE_DELTA = float(os.environ.get("RECOGNIZE_MIN_HANDSHAPE_DELTA", "0.025"))
RECOGNIZE_RIGID_SLIDE_PATH_MIN = float(os.environ.get("RECOGNIZE_RIGID_SLIDE_PATH_MIN", "0.20"))
RECOGNIZE_RIGID_SLIDE_EFFICIENCY = float(os.environ.get("RECOGNIZE_RIGID_SLIDE_EFFICIENCY", "0.88"))
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
_pose_lookup = None
_pose_lookup_lock = Lock()


def _configured_preload_texts() -> list[str]:
    raw = os.environ.get(
        "POSE_PRELOAD_TEXTS",
        "hello,please help me,good morning nice meet you,thank you for help",
    ).strip()
    if raw.lower() in ("", "0", "false", "no", "none"):
        return []
    seen = set()
    texts = []
    for item in raw.split(","):
        text = item.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            texts.append(text)
    return texts


POSE_PRELOAD_TEXTS = _configured_preload_texts()
_preload_status = {
    "configured": len(POSE_PRELOAD_TEXTS),
    "completed": 0,
    "failed": 0,
    "running": False,
    "finished": False,
    "last_error": None,
}

# --- Sign recognition lexicon (loaded at startup) ---
_lexicon_features: dict = {}   # gloss -> {"mean": np.ndarray, "sequence": np.ndarray}
_lexicon_features_lock = Lock()
_recognition_debug_lock = Lock()
_last_recognition_debug: dict = {
    "status": "not_run",
    "fallback_used": False,
}


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
LEXICON_ALIAS_FILE = os.environ.get("LEXICON_ALIAS_FILE") or os.path.join(LEXICON_DIR, "aliases.json")


def _load_gloss_resolver() -> GlossResolver:
    try:
        return GlossResolver.from_paths(LEXICON_DIR, LEXICON_ALIAS_FILE)
    except Exception as e:
        print(f"[Startup] Ignoring unusable alias file {LEXICON_ALIAS_FILE}: {e}")
        return GlossResolver.from_paths(LEXICON_DIR, None)


GLOSS_RESOLVER = _load_gloss_resolver()


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


def _cache_snapshot() -> dict:
    with _pose_cache_lock:
        return {
            "entries": len(_pose_cache),
            "maxEntries": POSE_CACHE_SIZE,
        }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _add_timing(timings: dict, key: str, started_at: float) -> None:
    timings[key] = round(timings.get(key, 0.0) + _elapsed_ms(started_at), 3)


@contextmanager
def _pose_library_output():
    if POSE_LIBRARY_LOGS:
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


class _ThreadSafePoseLookupMixin:
    def _init_threadsafe_cache(self, maxsize: int) -> None:
        from spoken_to_signed.gloss_to_pose.lookup.lru_cache import LRUCache

        self.cache = LRUCache(maxsize=maxsize)
        self._cache_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_pose(self, row):
        from pose_format import Pose

        pose_path = row["path"]
        with self._cache_lock:
            pose = self.cache.get(pose_path)
            if pose is None:
                pose = self.read_pose(pose_path)
                self.cache.set(pose_path, pose)
                self.cache_misses += 1
            else:
                self.cache_hits += 1

        frame_time = 1000 / pose.body.fps
        start_frame = math.floor(row["start"] // frame_time)
        end_frame = math.ceil(row["end"] // frame_time) if row["end"] > 0 else -1
        body = pose.body[start_frame:end_frame]
        # pose-format slices are NumPy views; concatenate/trim mutates bodies.
        # Copy here so a long-lived lookup cache never pollutes source poses.
        body.data = np.array(body.data, copy=True)
        body.confidence = np.array(body.confidence, copy=True)
        return Pose(pose.header, body)

    def cache_snapshot(self) -> dict:
        with self._cache_lock:
            return {
                "entries": len(self.cache.cache),
                "maxEntries": self.cache.maxsize,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            }


def _create_pose_lookup():
    from spoken_to_signed.gloss_to_pose.lookup.csv_lookup import CSVPoseLookup
    from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup

    class SafeFingerspellingPoseLookup(_ThreadSafePoseLookupMixin, FingerspellingPoseLookup):
        def __init__(self):
            super().__init__()
            self._init_threadsafe_cache(POSE_LOOKUP_CACHE_SIZE)

    class SafeCSVPoseLookup(_ThreadSafePoseLookupMixin, CSVPoseLookup):
        def __init__(self, directory: str, backup=None):
            super().__init__(directory=directory, backup=backup)
            self._init_threadsafe_cache(POSE_LOOKUP_CACHE_SIZE)

    return SafeCSVPoseLookup(LEXICON_DIR, backup=SafeFingerspellingPoseLookup())


def _get_pose_lookup():
    global _pose_lookup
    if _pose_lookup is None:
        with _pose_lookup_lock:
            if _pose_lookup is None:
                _pose_lookup = _create_pose_lookup()
    return _pose_lookup


def _pose_lookup_snapshot() -> dict:
    lookup = _pose_lookup
    if lookup is None:
        return {
            "entries": 0,
            "maxEntries": POSE_LOOKUP_CACHE_SIZE,
            "hits": 0,
            "misses": 0,
            "initialized": False,
        }
    snapshot = lookup.cache_snapshot()
    snapshot["initialized"] = True
    return snapshot

# ---------------------------------------------------------------------------
# Sign recognition helpers
# ---------------------------------------------------------------------------

def _plain_float_array(value, fill_value=np.nan) -> np.ndarray:
    """Return a regular float32 ndarray, replacing masked pose values first."""
    return np.asarray(np.ma.filled(value, fill_value), dtype=np.float32)


def _coerce_hand(lm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if lm is None:
        return None
    arr = _plain_float_array(lm)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        return None
    return arr


def _normalize_hand(lm: Optional[np.ndarray]) -> np.ndarray:
    """Center a hand at its wrist and normalize by max finger distance.

    Both the browser MediaPipe [0-1] space and the lexicon pixel space are
    reduced to a unit-scale, wrist-origin shape descriptor, making them
    comparable regardless of absolute coordinate system.
    """
    zeros = np.zeros((21, 3), dtype=np.float32)
    lm = _coerce_hand(lm)
    if lm is None:
        return zeros
    lm = lm.astype(np.float32, copy=True)
    lm -= lm[0]  # center at wrist (index 0)
    scale = np.linalg.norm(lm, axis=1).max()
    if scale > 1e-6:
        lm /= scale
    return lm


def _frame_feature(lh: Optional[np.ndarray], rh: Optional[np.ndarray]) -> np.ndarray:
    """126-dim feature vector: normalized left hand (63) + right hand (63)."""
    return np.concatenate([_normalize_hand(lh).flatten(), _normalize_hand(rh).flatten()])


def _valid_hand(lm: Optional[np.ndarray]) -> bool:
    return _coerce_hand(lm) is not None


def _hand_scale(lm: np.ndarray) -> float:
    lm = _plain_float_array(lm)
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
        lh_arr = _coerce_hand(lh)
        rh_arr = _coerce_hand(rh)
        lh_valid = lh_arr is not None
        rh_valid = rh_arr is not None
        if not lh_valid and not rh_valid:
            continue

        shape_feats.append(_frame_feature(lh_arr, rh_arr))
        left_wrists.append(lh_arr[0, :2] if lh_valid else [np.nan, np.nan])
        right_wrists.append(rh_arr[0, :2] if rh_valid else [np.nan, np.nan])
        visibility.append([1.0 if lh_valid else 0.0, 1.0 if rh_valid else 0.0])
        if lh_valid:
            scales.append(_hand_scale(lh_arr))
        if rh_valid:
            scales.append(_hand_scale(rh_arr))

    if not shape_feats:
        return None, None

    shape = np.stack(shape_feats).astype(np.float32)
    mean = np.asarray(shape.mean(axis=0), dtype=np.float32)
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
    sequence = np.asarray(np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
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
        if lh_sl is not None:
            lh_conf = _plain_float_array(pose.body.confidence[f, 0, lh_sl], 0.0)
            lh_candidate = _coerce_hand(pose.body.data[f, 0, lh_sl, :3])
            if lh_candidate is not None and float(lh_conf.mean()) > HAND_CONFIDENCE_MIN:
                lh_data = lh_candidate
        if rh_sl is not None:
            rh_conf = _plain_float_array(pose.body.confidence[f, 0, rh_sl], 0.0)
            rh_candidate = _coerce_hand(pose.body.data[f, 0, rh_sl, :3])
            if rh_candidate is not None and float(rh_conf.mean()) > HAND_CONFIDENCE_MIN:
                rh_data = rh_candidate
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


def _recognition_config_snapshot() -> dict:
    return {
        "top_k": RECOGNIZE_TOP_K,
        "sequence_length": RECOGNIZE_SEQUENCE_LENGTH,
        "temporal_weight": RECOGNIZE_TEMPORAL_WEIGHT,
        "dtw_band": RECOGNIZE_DTW_BAND,
        "confidence_min": RECOGNIZE_CONFIDENCE_MIN,
        "margin_min": RECOGNIZE_MARGIN_MIN,
        "high_confidence_min": RECOGNIZE_HIGH_CONFIDENCE_MIN,
        "high_confidence_margin_min": RECOGNIZE_HIGH_CONFIDENCE_MARGIN_MIN,
        "min_visible_frames": RECOGNIZE_MIN_VISIBLE_FRAMES,
        "min_visible_ratio": RECOGNIZE_MIN_VISIBLE_RATIO,
        "min_motion_path": RECOGNIZE_MIN_MOTION_PATH,
        "min_handshape_delta": RECOGNIZE_MIN_HANDSHAPE_DELTA,
        "rigid_slide_path_min": RECOGNIZE_RIGID_SLIDE_PATH_MIN,
        "rigid_slide_efficiency": RECOGNIZE_RIGID_SLIDE_EFFICIENCY,
        "hand_confidence_min": HAND_CONFIDENCE_MIN,
        "trajectory_feature_weight": TRAJECTORY_FEATURE_WEIGHT,
        "visibility_feature_weight": VISIBILITY_FEATURE_WEIGHT,
    }


def _recognition_index_snapshot() -> dict:
    with _lexicon_features_lock:
        classes_loaded = len(_lexicon_features)
        sample_glosses = sorted(_lexicon_features.keys())[:12]
    with _recognition_debug_lock:
        last = dict(_last_recognition_debug)
    return {
        "classes_loaded": classes_loaded,
        "index_path": os.path.join(LEXICON_DIR, "index.csv"),
        "lexicon_dir": LEXICON_DIR,
        "sample_glosses": sample_glosses,
        "config": _recognition_config_snapshot(),
        "last": last,
    }


def _hand_frame_stats(hand_frames: list) -> dict:
    total = len(hand_frames)
    visible = 0
    left_visible = 0
    right_visible = 0
    both_visible = 0
    paths = []
    displacements = []
    efficiencies = []
    shape_deltas = []
    for lh, rh in hand_frames:
        lh_valid = _coerce_hand(lh) is not None
        rh_valid = _coerce_hand(rh) is not None
        if lh_valid or rh_valid:
            visible += 1
        if lh_valid:
            left_visible += 1
        if rh_valid:
            right_visible += 1
        if lh_valid and rh_valid:
            both_visible += 1

    for side in (0, 1):
        hand_sequence = [_coerce_hand(frame[side]) for frame in hand_frames]
        hand_sequence = [hand for hand in hand_sequence if hand is not None]
        if len(hand_sequence) < 2:
            continue

        scales = [_hand_scale(hand) for hand in hand_sequence]
        scale_values = [scale for scale in scales if scale > 1e-6 and np.isfinite(scale)]
        scale = float(np.median(scale_values)) if scale_values else 1.0
        if scale <= 1e-6 or not np.isfinite(scale):
            scale = 1.0

        wrists = np.array([hand[0, :2] for hand in hand_sequence], dtype=np.float32)
        step_distances = np.linalg.norm(np.diff(wrists, axis=0), axis=1) / scale
        path = float(np.sum(step_distances))
        displacement = float(np.linalg.norm(wrists[-1] - wrists[0]) / scale)
        efficiency = displacement / path if path > 1e-6 else 0.0
        normalized_shapes = np.array([_normalize_hand(hand).flatten() for hand in hand_sequence], dtype=np.float32)
        shape_delta = float(np.mean(np.linalg.norm(np.diff(normalized_shapes, axis=0), axis=1)))

        paths.append(path)
        displacements.append(displacement)
        efficiencies.append(efficiency)
        shape_deltas.append(shape_delta)

    return {
        "frames_received": total,
        "visible_frames": visible,
        "visibility_ratio": round(visible / total, 4) if total else 0.0,
        "left_visible_frames": left_visible,
        "right_visible_frames": right_visible,
        "both_visible_frames": both_visible,
        "active_hand_sides": len(paths),
        "motion_path": round(max(paths), 4) if paths else 0.0,
        "motion_displacement": round(max(displacements), 4) if displacements else 0.0,
        "motion_efficiency": round(max(efficiencies), 4) if efficiencies else 0.0,
        "handshape_delta": round(max(shape_deltas), 4) if shape_deltas else 0.0,
    }


def _not_confident_response(reason: str, debug: dict, candidates: Optional[list] = None, best: Optional[dict] = None) -> dict:
    candidates = candidates or []
    best = best or (candidates[0] if candidates else None)
    response = {
        "gloss": "",
        "recognized_gloss": best.get("gloss", "") if best else "",
        "confidence": best.get("confidence", 0.0) if best else 0.0,
        "mean_confidence": best.get("mean_confidence", 0.0) if best else 0.0,
        "temporal_confidence": best.get("temporal_confidence", 0.0) if best else 0.0,
        "accepted": False,
        "reason": reason,
        "fallback_used": False,
        "candidates": candidates[:5],
        "debug": debug,
    }
    with _recognition_debug_lock:
        _last_recognition_debug.clear()
        _last_recognition_debug.update({
            "status": "rejected",
            "reason": reason,
            "fallback_used": False,
            "top_candidates": candidates[:5],
            **debug,
        })
    return response


def _resolve_pose_text(text: str) -> str:
    resolved_text, resolutions = GLOSS_RESOLVER.resolve_text(text)
    changed = resolved_text != text.strip().lower()
    aliased = [item for item in resolutions if item.method in ("alias", "variant", "exact")]
    if changed and aliased:
        preview = ", ".join(f"{item.source}->{item.target}" for item in aliased[:6])
        print(f"[Gloss] Resolved before fingerspelling: {preview}")
    return resolved_text


def generate_pose(text: str, retries: int = 2) -> bytes:
    pose_bytes, _ = generate_pose_detailed(text, retries=retries)
    return pose_bytes


def generate_pose_detailed(text: str, retries: int = 2) -> tuple[bytes, dict]:
    """Generate a .pose binary and return per-stage latency timings.

    The default backend uses the spoken-to-signed Python API directly and keeps
    the lexicon lookup cache warm across requests. `POSE_BACKEND=cli` keeps the
    old subprocess path available as a rollback switch.
    """
    timings = {
        "backend": POSE_BACKEND if POSE_BACKEND == "cli" else "library",
        "cache": "miss",
    }
    total_started = time.perf_counter()

    started = time.perf_counter()
    resolved_text = _resolve_pose_text(text)
    _add_timing(timings, "alias_resolution", started)

    started = time.perf_counter()
    cached = _cache_get(resolved_text)
    _add_timing(timings, "pose_cache_lookup", started)
    if cached is not None:
        timings["cache"] = "hit"
        timings["bytes"] = len(cached)
        timings["pose_total"] = _elapsed_ms(total_started)
        return cached, timings

    if POSE_BACKEND == "cli":
        pose_bytes, cli_timings = _generate_pose_cli_processed(resolved_text, retries=retries)
        timings.update(cli_timings)
    else:
        try:
            pose_bytes, library_timings = _generate_pose_library_processed(resolved_text)
            timings.update(library_timings)
        except Exception as e:
            print(f"[Pose] In-process backend failed ({str(e)[:100]}); falling back to CLI")
            fallback_bytes, cli_timings = _generate_pose_cli_processed(resolved_text, retries=retries)
            timings.update(cli_timings)
            timings["backend"] = "cli_fallback"
            pose_bytes = fallback_bytes

    timings["bytes"] = len(pose_bytes)
    timings["pose_total"] = _elapsed_ms(total_started)
    _cache_set(resolved_text, pose_bytes)
    return pose_bytes, timings


def _generate_pose_library_processed(resolved_text: str) -> tuple[bytes, dict]:
    timings: dict = {"backend": "library"}
    started = time.perf_counter()
    from spoken_to_signed.gloss_to_pose import PoseResult, concatenate_poses
    from spoken_to_signed.text_to_gloss.simple import text_to_gloss
    _add_timing(timings, "pose_library_import", started)

    started = time.perf_counter()
    lookup = _get_pose_lookup()
    _add_timing(timings, "pose_lookup_init", started)

    with _pose_library_output():
        started = time.perf_counter()
        sentences = text_to_gloss(text=resolved_text, language="en")
        _add_timing(timings, "simple_gloss", started)

        sentence_results = []
        for sentence in sentences:
            poses = []
            for item in sentence:
                if not item.word:
                    continue
                started = time.perf_counter()
                try:
                    poses.append(lookup.lookup(item.word, item.gloss, "en", "ase").pose)
                except FileNotFoundError as e:
                    print(e)
                _add_timing(timings, "pose_loading", started)

            if not poses:
                gloss_sequence = " ".join([f"{item.word}/{item.gloss}" for item in sentence])
                raise Exception(f"No poses found for {gloss_sequence}")

            started = time.perf_counter()
            sentence_results.append(PoseResult(pose=concatenate_poses(poses)))
            _add_timing(timings, "motion_generation", started)

        if len(sentence_results) == 1:
            pose = sentence_results[0].pose
        else:
            started = time.perf_counter()
            pose = concatenate_poses([result.pose for result in sentence_results], trim=False)
            _add_timing(timings, "motion_generation", started)

    pose, post_timings = _postprocess_pose_object(pose)
    timings.update(post_timings)
    started = time.perf_counter()
    buf = io.BytesIO()
    pose.write(buf)
    _add_timing(timings, "pose_serialization", started)
    return buf.getvalue(), timings


def _generate_pose_cli_processed(resolved_text: str, retries: int = 2) -> tuple[bytes, dict]:
    """Run the legacy CLI path and return processed bytes plus timings."""
    timings: dict = {"backend": "cli"}
    last_err = None
    for attempt in range(retries + 1):
        fd, out_path = tempfile.mkstemp(suffix=".pose")
        os.close(fd)
        try:
            started = time.perf_counter()
            result = subprocess.run(
                [
                    CLI,
                    "--text", resolved_text,
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
            _add_timing(timings, "subprocess_generation", started)
            if result.returncode != 0:
                raise RuntimeError(f"text_to_gloss_to_pose failed: {result.stderr}")
            started = time.perf_counter()
            with open(out_path, "rb") as f:
                raw_bytes = f.read()
            _add_timing(timings, "pose_file_read", started)
            pose_bytes, post_timings = _postprocess_pose_bytes(raw_bytes)
            timings.update(post_timings)
            return pose_bytes, timings
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
    processed, _ = _postprocess_pose_bytes(pose_bytes)
    return processed


def _postprocess_pose_bytes(pose_bytes: bytes) -> tuple[bytes, dict]:
    try:
        from pose_format import Pose

        started = time.perf_counter()
        pose = Pose.read(pose_bytes)
        timings = {"pose_parse": _elapsed_ms(started)}
        pose, post_timings = _postprocess_pose_object(pose)
        timings.update(post_timings)
        started = time.perf_counter()
        buf = io.BytesIO()
        pose.write(buf)
        _add_timing(timings, "pose_serialization", started)
        return buf.getvalue(), timings
    except Exception as e:
        print(f"[Pose] Postprocess skipped: {e}")
        return pose_bytes, {"pose_processing": 0.0, "pose_serialization": 0.0}


def _postprocess_pose_object(pose) -> tuple[object, dict]:
    """Apply runtime cleanup passes to an in-memory pose."""
    timings: dict = {}
    processing_started = time.perf_counter()
    if STABILIZE_BODY_SIZE:
        started = time.perf_counter()
        _stabilize_body_size(pose)
        _add_timing(timings, "body_stabilization", started)
    if STABILIZE_FACE_SIZE:
        started = time.perf_counter()
        _stabilize_face_size(pose)
        _add_timing(timings, "face_stabilization", started)
    # Simplify after stabilization (which needs the dense mesh) but before the
    # Y-shift, which operates on whatever components the pose currently has.
    if SIMPLE_FACE:
        started = time.perf_counter()
        pose = _simplify_face(pose)
        _add_timing(timings, "face_simplification", started)
    if AVATAR_Y_OFFSET:
        started = time.perf_counter()
        _shift_avatar_y(pose, AVATAR_Y_OFFSET)
        _add_timing(timings, "avatar_shift", started)
    if SIGN_SPEED != 1.0:
        pose.body.fps = pose.body.fps * SIGN_SPEED

    timings["pose_processing"] = _elapsed_ms(processing_started)
    return pose, timings


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
    pose_data = pose.body.data[:, :, pose_slice, :] if pose_slice is not None else None
    pose_confidence = pose.body.confidence[:, :, pose_slice] if pose_slice is not None else None

    visible = confidence > 0
    xy_all = data[:, :, :, :2]
    finite_xy = np.isfinite(xy_all).all(axis=3)
    nonzero_xy = np.abs(xy_all).sum(axis=3) > 0
    valid_visible = visible & finite_xy & nonzero_xy
    confidence[visible & ~valid_visible] = 0
    visible = valid_visible

    eligible = visible.sum(axis=2) >= 20
    if not eligible.any():
        return

    block = data[eligible].copy()
    visible_block = visible[eligible]
    masked_xy = np.where(visible_block[:, :, None], block[:, :, :2], np.nan)

    with np.errstate(all="ignore"):
        min_xy = np.nanpercentile(masked_xy, 2, axis=1)
        max_xy = np.nanpercentile(masked_xy, 98, axis=1)
    face_size = max_xy - min_xy
    face_width = face_size[:, 0]
    face_height = face_size[:, 1]
    valid_face = (
        np.isfinite(face_width)
        & np.isfinite(face_height)
        & (face_width > 1)
        & (face_height > 1)
    )
    if not valid_face.any():
        return

    valid_block = block[valid_face].copy()
    valid_mask = visible_block[valid_face]
    centers = ((min_xy[valid_face] + max_xy[valid_face]) / 2).astype(np.float32)

    scale_x = np.ones(len(valid_block), dtype=np.float32)
    scale_y = np.ones(len(valid_block), dtype=np.float32)
    if TARGET_FACE_WIDTH > 0:
        scale_x = np.clip(TARGET_FACE_WIDTH / face_width[valid_face], 0.65, 1.45).astype(np.float32)
    if TARGET_FACE_HEIGHT > 0:
        scale_y = np.clip(TARGET_FACE_HEIGHT / face_height[valid_face], 0.65, 1.45).astype(np.float32)
    finite_scale = np.isfinite(scale_x) & np.isfinite(scale_y)
    if not finite_scale.any():
        return

    valid_block = valid_block[finite_scale].copy()
    valid_mask = valid_mask[finite_scale]
    centers = centers[finite_scale]
    scale_x = scale_x[finite_scale]
    scale_y = scale_y[finite_scale]

    valid_block[:, :, 0] = np.where(
        valid_mask,
        (valid_block[:, :, 0] - centers[:, None, 0]) * scale_x[:, None] + centers[:, None, 0],
        valid_block[:, :, 0],
    )
    valid_block[:, :, 1] = np.where(
        valid_mask,
        (valid_block[:, :, 1] - centers[:, None, 1]) * scale_y[:, None] + centers[:, None, 1],
        valid_block[:, :, 1],
    )
    valid_block[:, :, 2] = np.where(
        valid_mask,
        valid_block[:, :, 2] * ((scale_x + scale_y) / 2)[:, None],
        valid_block[:, :, 2],
    )

    eligible_coords = np.column_stack(np.nonzero(eligible))
    valid_coords = eligible_coords[valid_face][finite_scale]

    if (
        pose_data is not None
        and pose_confidence is not None
        and left_shoulder_index is not None
        and right_shoulder_index is not None
    ):
        frame_indices = valid_coords[:, 0]
        person_indices = valid_coords[:, 1]
        left_conf = pose_confidence[frame_indices, person_indices, left_shoulder_index]
        right_conf = pose_confidence[frame_indices, person_indices, right_shoulder_index]
        left_shoulder = pose_data[frame_indices, person_indices, left_shoulder_index, :2]
        right_shoulder = pose_data[frame_indices, person_indices, right_shoulder_index, :2]
        shoulder_valid = (
            (left_conf > 0)
            & (right_conf > 0)
            & np.isfinite(left_shoulder).all(axis=1)
            & np.isfinite(right_shoulder).all(axis=1)
        )
        if shoulder_valid.any():
            shoulder_center = (left_shoulder[shoulder_valid] + right_shoulder[shoulder_valid]) / 2
            target_center = shoulder_center + np.array(
                [TARGET_FACE_SHOULDER_X, TARGET_FACE_SHOULDER_Y],
                dtype=np.float32,
            )
            shift = target_center - centers[shoulder_valid]
            valid_block[shoulder_valid, :, :2] = np.where(
                valid_mask[shoulder_valid, :, None],
                valid_block[shoulder_valid, :, :2] + shift[:, None, :],
                valid_block[shoulder_valid, :, :2],
            )

    valid_indices = np.flatnonzero(valid_face)[finite_scale]
    block[valid_indices] = valid_block
    data[eligible] = block


def _preload_pose_cache() -> None:
    _preload_status["running"] = True
    _preload_status["finished"] = False
    for text in POSE_PRELOAD_TEXTS:
        try:
            generate_pose(text)
            _preload_status["completed"] += 1
        except Exception as e:
            _preload_status["failed"] += 1
            _preload_status["last_error"] = str(e)[:160]
            print(f"[Preload] Failed to cache {text!r}: {e}")
    _preload_status["running"] = False
    _preload_status["finished"] = True
    if POSE_PRELOAD_TEXTS:
        print(
            f"[Preload] Cached {_preload_status['completed']}/{len(POSE_PRELOAD_TEXTS)} "
            f"configured phrase(s)"
        )


def _start_pose_preload() -> None:
    if not POSE_PRELOAD_TEXTS:
        _preload_status["finished"] = True
        return
    Thread(target=_preload_pose_cache, daemon=True).start()


@app.on_event("startup")
async def startup():
    print(f"[Startup] ASL Pose Server listening on port {PORT}")
    print(f"[Startup] text_to_gloss_to_pose CLI: {CLI}")
    print(f"[Startup] Lexicon dir: {LEXICON_DIR}")
    print(f"[Startup] Pose backend: {POSE_BACKEND} (lookup cache={POSE_LOOKUP_CACHE_SIZE})")
    print(
        f"[Startup] Gloss resolver: {len(GLOSS_RESOLVER.lexicon_glosses)} gloss(es), "
        f"{len(GLOSS_RESOLVER.aliases)} alias(es) from {LEXICON_ALIAS_FILE}"
    )
    try:
        _, timings = generate_pose_detailed("hello")
        print(f"[Warmup] Pose model ready ({timings.get('pose_total', 0):.1f}ms)")
    except Exception as e:
        print(f"[Warmup] Warning: {e}")
    # Load lexicon features for sign recognition (runs in the startup thread).
    _load_lexicon_features()
    _start_pose_preload()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pose": {
            "backend": POSE_BACKEND if POSE_BACKEND == "cli" else "library",
            "signSpeed": SIGN_SPEED,
            "responseCache": _cache_snapshot(),
            "lookupCache": _pose_lookup_snapshot(),
            "preload": dict(_preload_status),
        },
        "lexicon": {
            "dir": LEXICON_DIR,
            "glosses": len(GLOSS_RESOLVER.lexicon_glosses),
            "aliases": len(GLOSS_RESOLVER.aliases),
        },
        "recognition": _recognition_index_snapshot(),
    }


@app.get("/recognize/debug")
def recognize_debug():
    return _recognition_index_snapshot()


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

    stats = _hand_frame_stats(hand_frames)
    debug = {
        **stats,
        "classes_loaded": n_signs,
        "index_path": os.path.join(LEXICON_DIR, "index.csv"),
        "fallback_used": False,
        "accepted": False,
        "reason": None,
        "candidates_considered": 0,
        "top_margin": 0.0,
    }

    if stats["visible_frames"] < RECOGNIZE_MIN_VISIBLE_FRAMES:
        reason = (
            f"insufficient_hand_visibility: {stats['visible_frames']} visible frame(s), "
            f"need {RECOGNIZE_MIN_VISIBLE_FRAMES}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    if stats["visibility_ratio"] < RECOGNIZE_MIN_VISIBLE_RATIO:
        reason = (
            f"insufficient_hand_visibility_ratio: {stats['visibility_ratio']:.2f}, "
            f"need {RECOGNIZE_MIN_VISIBLE_RATIO:.2f}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    if (
        stats["motion_path"] < RECOGNIZE_MIN_MOTION_PATH
        and stats["handshape_delta"] < RECOGNIZE_MIN_HANDSHAPE_DELTA
    ):
        reason = (
            f"insufficient_motion: path {stats['motion_path']:.2f}, "
            f"handshape_delta {stats['handshape_delta']:.3f}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    if (
        stats["active_hand_sides"] == 1
        and stats["motion_path"] >= RECOGNIZE_RIGID_SLIDE_PATH_MIN
        and stats["motion_efficiency"] >= RECOGNIZE_RIGID_SLIDE_EFFICIENCY
        and stats["handshape_delta"] < RECOGNIZE_MIN_HANDSHAPE_DELTA
    ):
        reason = (
            f"rigid_slide_rejected: efficiency {stats['motion_efficiency']:.2f}, "
            f"handshape_delta {stats['handshape_delta']:.3f}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    query_mean, query_sequence = _hand_frames_to_features(hand_frames)
    if query_mean is None or query_sequence is None:
        reason = "no_usable_hand_features"
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    candidates = []
    with _lexicon_features_lock:
        for gloss, ref in _lexicon_features.items():
            ref_mean = _plain_float_array(ref["mean"], 0.0)
            if ref_mean.shape != query_mean.shape:
                continue
            mean_score = float(np.dot(query_mean, ref_mean))
            if not np.isfinite(mean_score):
                continue
            candidates.append((mean_score, gloss, ref))

    if not candidates:
        reason = "no_reference_candidates"
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    candidates.sort(key=lambda item: item[0], reverse=True)
    reranked = []

    for mean_score, gloss, ref in candidates[:RECOGNIZE_TOP_K]:
        ref_sequence = _plain_float_array(ref["sequence"], 0.0)
        if ref_sequence.shape != query_sequence.shape:
            continue
        temporal_score = _temporal_similarity(query_sequence, ref_sequence)
        combined_score = (
            (1.0 - RECOGNIZE_TEMPORAL_WEIGHT) * mean_score
            + RECOGNIZE_TEMPORAL_WEIGHT * temporal_score
        )
        if not np.isfinite(combined_score):
            continue
        reranked.append({
            "gloss": gloss,
            "confidence": round(float(combined_score), 4),
            "mean_confidence": round(float(mean_score), 4),
            "temporal_confidence": round(float(temporal_score), 4),
        })

    reranked.sort(key=lambda item: item["confidence"], reverse=True)
    debug["candidates_considered"] = min(RECOGNIZE_TOP_K, len(candidates))

    if not reranked:
        reason = "no_temporal_candidates"
        debug["reason"] = reason
        return _not_confident_response(reason, debug)

    best = reranked[0]
    runner_up = reranked[1] if len(reranked) > 1 else None
    top_margin = best["confidence"] - (runner_up["confidence"] if runner_up else 0.0)
    debug["top_margin"] = round(float(top_margin), 4)
    debug["top_candidates"] = reranked[:5]

    if best["confidence"] < RECOGNIZE_CONFIDENCE_MIN:
        reason = (
            f"low_confidence: {best['confidence']:.2f}, "
            f"need {RECOGNIZE_CONFIDENCE_MIN:.2f}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug, reranked[:5], best)

    required_margin = (
        RECOGNIZE_HIGH_CONFIDENCE_MARGIN_MIN
        if best["confidence"] >= RECOGNIZE_HIGH_CONFIDENCE_MIN
        else RECOGNIZE_MARGIN_MIN
    )
    debug["required_margin"] = round(float(required_margin), 4)

    if runner_up and top_margin < required_margin:
        reason = (
            f"ambiguous_match: margin {top_margin:.2f}, "
            f"need {required_margin:.2f}"
        )
        debug["reason"] = reason
        return _not_confident_response(reason, debug, reranked[:5], best)

    debug["accepted"] = True
    debug["reason"] = "accepted"
    with _recognition_debug_lock:
        _last_recognition_debug.clear()
        _last_recognition_debug.update({
            "status": "accepted",
            "reason": "accepted",
            "fallback_used": False,
            **debug,
        })

    return {
        "gloss": best["gloss"],
        "recognized_gloss": best["gloss"],
        "confidence": best["confidence"],
        "mean_confidence": best["mean_confidence"],
        "temporal_confidence": best["temporal_confidence"],
        "accepted": True,
        "reason": "accepted",
        "fallback_used": False,
        "candidates": reranked[:5],
        "candidates_considered": min(RECOGNIZE_TOP_K, len(candidates)),
        "debug": debug,
    }


@app.post("/pose")
def pose(req: PoseRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        pose_bytes, timings = generate_pose_detailed(text)
    except Exception as e:
        print(f"[Pose] Error generating pose: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(
        content=pose_bytes,
        media_type="application/octet-stream",
        headers={
            "X-DeepSign-Pose-Cache": str(timings.get("cache", "unknown")),
            "X-DeepSign-Pose-Backend": str(timings.get("backend", "unknown")),
            "X-DeepSign-Pose-Timings": json.dumps(timings, separators=(",", ":")),
        },
    )
