from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Any

from asl.planner import plan_asl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

PORT = int(os.environ.get("PORT", "8000"))


class MotionRequest(BaseModel):
    text: str
    generate_pose: bool = Field(default=True)
    source: str = Field(default="speech")


class PoseRequest(BaseModel):
    text: str


def _resolve_cli() -> str | None:
    bin_dir = os.path.dirname(sys.executable)
    for name in ("text_to_gloss_to_pose", "text_to_gloss_to_pose.exe"):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("text_to_gloss_to_pose")


def _resolve_lexicon_dir() -> str | None:
    override = os.environ.get("LEXICON_DIR")
    if override:
        return override
    try:
        import spoken_to_signed
    except Exception:
        return None
    return os.path.join(
        os.path.dirname(spoken_to_signed.__file__),
        "assets",
        "fingerspelling_lexicon",
    )


CLI = _resolve_cli()
LEXICON_DIR = _resolve_lexicon_dir()

app = FastAPI(title="DeepSign ASL Motion Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=512)
def generate_pose(text: str) -> bytes:
    if not CLI:
        raise RuntimeError("text_to_gloss_to_pose CLI is not installed")
    if not LEXICON_DIR:
        raise RuntimeError("No lexicon directory is available")

    out_path = tempfile.mktemp(suffix=".pose")
    try:
        result = subprocess.run(
            [
                CLI,
                "--text",
                text,
                "--glosser",
                "simple",
                "--spoken-language",
                "en",
                "--signed-language",
                "ase",
                "--lexicon",
                LEXICON_DIR,
                "--pose",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("POSE_TIMEOUT_SECONDS", "30")),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "text_to_gloss_to_pose failed")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def motion_clip_for_unit(unit: dict[str, Any], fallback_text: str, generate: bool) -> dict[str, Any]:
    hands = unit.get("hands") or {
        "pattern": "one_handed",
        "active": ["dominant"],
    }

    if unit["type"] == "sign":
        text = " ".join(unit.get("english") or [unit["gloss"].lower()])
        sign_id = unit["gloss"]
    elif unit["type"] == "fingerspell":
        text = unit["text"]
        sign_id = f"FS:{unit['text'].upper()}"
    else:
        return {
            "kind": "caption",
            "text": unit.get("text", fallback_text),
            "reason": unit.get("reason", "no_motion_available"),
        }

    clip = {
        "kind": "pose",
        "id": sign_id,
        "text": text,
        "format": "pose-v0.2",
        "source": "spoken-to-signed-translation",
        "hands": hands,
        "requiresTwoHands": len(hands.get("active", [])) == 2,
        "motionPattern": hands.get("pattern", "one_handed"),
    }

    if not generate:
        clip["status"] = "planned"
        return clip

    try:
        pose_bytes = generate_pose(text)
        clip["status"] = "ready"
        clip["encoding"] = "base64"
        clip["mime"] = "application/octet-stream"
        clip["data"] = base64.b64encode(pose_bytes).decode("ascii")
    except Exception as exc:
        clip["status"] = "unavailable"
        clip["error"] = str(exc)
    return clip


@app.on_event("startup")
async def startup():
    print(f"[Startup] DeepSign ASL Motion Server listening on port {PORT}")
    print(f"[Startup] text_to_gloss_to_pose CLI: {CLI or 'not installed'}")
    print(f"[Startup] Lexicon dir: {LEXICON_DIR or 'not available'}")
    if CLI and LEXICON_DIR and os.environ.get("POSE_WARMUP", "1") == "1":
        try:
            generate_pose("hello")
            print("[Warmup] Pose generator ready")
        except Exception as exc:
            print(f"[Warmup] Pose generator unavailable: {exc}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "poseGenerator": bool(CLI and LEXICON_DIR),
        "cli": CLI,
        "lexiconDir": LEXICON_DIR,
    }


@app.post("/plan")
async def plan(req: PoseRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    return plan_asl(text)


@app.post("/motion")
async def motion(req: MotionRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    plan = plan_asl(text)
    clips = [
        motion_clip_for_unit(unit, text, req.generate_pose)
        for unit in plan["units"]
    ]
    warnings = [
        clip["error"]
        for clip in clips
        if clip.get("status") == "unavailable" and clip.get("error")
    ]
    return {
        "source": req.source,
        "plan": plan,
        "clips": clips,
        "warnings": warnings,
    }


@app.post("/pose")
async def pose(req: PoseRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        pose_bytes = generate_pose(text)
    except Exception as exc:
        print(f"[Pose] Error generating pose: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(content=pose_bytes, media_type="application/octet-stream")
