# DeepSign

Real-time English speech to an ASL-oriented signing avatar experience for Meta Quest 3.

This repo is now structured as a runnable MVP plus the right seams for a serious
signing system:

```text
Quest mic or typed text
-> Node.js realtime gateway
-> Deepgram Nova-3 STT, when configured
-> Python FastAPI ASL motion server
-> ASL planner
-> pose fallback / future SignAvatars motion library
-> browser display client
```

## What works today

- Browser display client optimized for Quest-sized screens.
- Typed phrase testing without Deepgram.
- WebSocket display channel for transcripts, planner output, and motion clips.
- `/audio` WebSocket for raw PCM16 speech streaming to Deepgram.
- AudioWorklet mic capture at 16 kHz PCM16.
- Python ASL planner with curated phrase, lexical-sign, fingerspelling, and caption fallbacks.
- Two-handed planning metadata for symmetrical, alternating symmetrical, and asymmetrical signs.
- Legacy `.pose` generation through ZurichNLP `spoken-to-signed-translation` when installed.
- Caption-only fallback when Deepgram or the pose generator is unavailable.

## What is intentionally still a roadmap item

- Fluency-validated ASL grammar.
- A Deaf-reviewed phrase/sign library.
- SignAvatars SMPL-X to VRM retargeting.
- Production Quest native app packaging.
- A high-quality signing-specific VRM avatar.

## Install

### Node

```bash
npm install
```

### Python

```bash
cd python
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ..
```

The Python install pulls `spoken-to-signed-translation` from GitHub. If you only
want typed planner/caption mode at first, you can temporarily install just:

```bash
cd python
python3 -m venv .venv
.venv/bin/python -m pip install fastapi "uvicorn[standard]"
cd ..
```

## Configure

```bash
cp .env.example .env
```

Set `DEEPGRAM_API_KEY` to enable live speech. Without it, the app still runs in
typed-text mode.

Useful environment variables:

- `DEEPGRAM_API_KEY`: enables `/audio` speech streaming.
- `DEEPGRAM_MODEL`: defaults to `nova-3`.
- `DEEPGRAM_ENDPOINTING_MS`: defaults to `300`.
- `POSE_SERVER_URL`: defaults to `http://localhost:8000`.
- `LEXICON_DIR`: optional `spoken-to-signed-translation` lexicon directory.
- `MAX_WORDS_PER_SIGN_JOB`: defaults to `8`.

## Run

Terminal 1:

```bash
cd python
.venv/bin/uvicorn server:app --port 8000
```

Terminal 2:

```bash
npm start
```

Open:

```text
http://localhost:3000
```

On Quest 3, use your machine's LAN IP:

```text
http://YOUR_LOCAL_IP:3000
```

## Test without a microphone

Use the text box in the browser, or:

```bash
curl -X POST http://localhost:3000/api/sign \
  -H "Content-Type: application/json" \
  -d '{"text":"I need help"}'
```

## Test the Python planner

```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"text":"What is your name"}'
```

## SignAvatars integration plan

SignAvatars should be used as the motion foundation, not as a direct runtime
dependency in the Quest browser.

Recommended pipeline:

```text
SignAvatars SMPL-X / MANO annotations
-> offline retargeting to your chosen VRM avatar
-> export per-sign or per-phrase clips as VRMA, glTF animation, or compact JSON/binary
-> index clips by gloss / phrase / HamNoSys
-> Python motion server returns clip IDs or clip bytes
-> Quest browser only blends and renders
```

Add this as a new Python module later:

```text
python/motion_library/
  manifest.json
  clips/
    HELP.vrma
    THANK-YOU.vrma
    ...
  retarget_signavatars.py
```

The ASL planner already returns stable units like `SIGN HELP` and
`FS H-E-L-L-O`, so replacing generated `.pose` clips with curated SignAvatars
motion clips is a contained change.

Each sign unit also includes a `hands` object:

```json
{
  "pattern": "asymmetrical",
  "active": ["dominant", "non_dominant"],
  "dominant": { "role": "articulator" },
  "nonDominant": { "role": "support", "motion": "hold" }
}
```

Use `pattern: "symmetrical"` for mirrored two-hand signs like `WORK` or
`SCHOOL`, `alternating: true` for alternating two-hand signs like `SIGN`, and
`pattern: "asymmetrical"` for signs where the dominant hand articulates against
a stable support or target hand, such as `HELP`, `NAME`, or `LEARN`.

## Reality check

This project can become a strong assistive or educational prototype, but
arbitrary English-to-ASL translation is not solved by wiring STT to a motion
model. Treat the current planner as a scaffold. Build a curated ASL phrase
library with fluent Deaf signers reviewing both grammar and avatar motion.
