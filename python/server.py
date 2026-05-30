import io
import csv
import os
import shutil
import subprocess
import sys
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

PORT = int(os.environ.get("PORT", "8000"))

# Playback speed multiplier for the signing avatar. pose-viewer animates at the
# pose's stored fps, so raising fps makes the avatar sign faster without
# dropping any motion. 1.0 = native speed; 1.6 ≈ 60% faster. Tune via env.
SIGN_SPEED = float(os.environ.get("SIGN_SPEED", "1.6"))


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
            return _apply_speed(f.read())
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _apply_speed(pose_bytes: bytes) -> bytes:
    """Speed up the avatar by raising the pose's stored fps. pose-viewer plays
    back at body.fps, so a higher value plays the same frames in less time.
    """
    if SIGN_SPEED == 1.0:
        return pose_bytes
    try:
        from pose_format import Pose

        pose = Pose.read(pose_bytes)
        pose.body.fps = pose.body.fps * SIGN_SPEED
        buf = io.BytesIO()
        pose.write(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[Pose] Speed-up skipped: {e}")
        return pose_bytes


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
