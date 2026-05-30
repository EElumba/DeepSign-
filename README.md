# ASL Pose Avatar MVP

Real-time speech → ASL stick-figure signer. You speak into a Ray-Ban Meta glasses
mic, Deepgram transcribes it, a Python server converts the text into a `.pose`
skeleton file via [`spoken-to-signed-translation`](https://github.com/ZurichNLP/spoken-to-signed-translation),
and the [`pose-viewer`](https://www.npmjs.com/package/pose-viewer) web component
animates the signer in the glasses browser.

```
Mic → Node /audio WS → Deepgram Nova-3 → Node POST :8000/pose
    → Python FastAPI (text_to_gloss_to_pose) → .pose binary
    → Node broadcasts ArrayBuffer over WS → pose-viewer renders the signer
```

## Prerequisites

- Node.js 18+
- Python 3.10+
- pip

## Install

```bash
# Node dependencies
npm install

# Python dependencies (in a virtual environment)
cd python
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ..
```

**Note on Python install time:** `spoken-to-signed-translation` installs several
NLP dependencies. The first install takes a few minutes. Subsequent starts are instant.

**Signing approach:** This MVP doesn't ship a full ASL word lexicon, so it uses
the fingerspelling lexicon bundled with `spoken-to-signed` — every English phrase
is signed letter-by-letter in ASL. To use real word-level signs, point the
`LEXICON_DIR` env var at a lexicon directory (a folder containing `index.csv`).

## Configure

```bash
cp .env.example .env
# Add your Deepgram API key to .env
```

## Run — two terminals required

**Terminal 1 — Python pose server:**

```bash
cd python
.venv/bin/uvicorn server:app --port 8000
```

Wait for: `Application startup complete.` (the first request is pre-warmed on startup).

**Terminal 2 — Node server:**

```bash
npm start
```

## Open on Meta glasses

```
http://YOUR_LOCAL_IP:3000
```

Grant mic permission, tap **Start**, and speak — the avatar signs.

## Test the Python server independently

```bash
curl -X POST http://localhost:8000/pose \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello nice to meet you"}' \
  --output test.pose
ls -la test.pose  # Should be a non-zero binary file
```
