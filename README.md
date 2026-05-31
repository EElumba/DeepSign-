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

**Signing approach:** Out of the box, `spoken-to-signed` only ships a
fingerspelling lexicon, so every word is spelled letter-by-letter (slow, one
hand). To get real **two-handed word-level ASL**, build a WLASL pose lexicon
(see [Build a two-handed ASL lexicon](#build-a-two-handed-asl-lexicon-wlasl--mediapipe)
below). The generated WLASL videos/pose files are ignored local assets, so a
fresh fork that only installs requirements will not have them yet. When
`python/lexicon_wlasl/index.csv` has entries that point at real `.pose` files,
the server uses it automatically; any word not in the lexicon still falls back
to fingerspelling. You can also point `LEXICON_DIR` at any lexicon directory (a
folder containing `index.csv`) to override.

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

## Open on Meta Ray-Ban Display glasses (Web App)

Meta Ray-Ban **Display** glasses run standalone Web Apps loaded by URL through
the Meta AI app. Per Meta's [Web Apps docs](https://wearables.developer.meta.com/docs/develop/webapps),
**Web Apps cannot access the microphone or camera** (those need the native
Device Access Toolkit). So the app is split into two roles:

| Page | Where it runs | Purpose |
| --- | --- | --- |
| `/speak` (or `/`) | Phone/laptop browser | Captures the mic, streams audio → Deepgram |
| `/glasses` | Meta Ray-Ban Display | Display-only: renders the signing avatar + captions |

Both connect to the same Node WebSocket server, so speaking on the phone makes
the glasses avatar sign — the glasses are the **receiver/display**.

The `/glasses` page follows the platform constraints: fixed **600×600** viewport,
no scrolling, black background (renders transparent on the waveguide) with
bright high-contrast UI, and **arrow-key/Enter** focusable controls (Neural Band
/ captouch). It auto-connects and auto-reconnects — no mic prompt.

### Load it on the glasses

1. Deploy over **HTTPS** (Railway already does this).
2. In the **Meta AI app** → Settings → App Info, tap the version number **5×** to enable **Developer Mode**.
3. In the Meta AI app → your **Display Glasses** → **App connections → Web apps → Add a web app**, enter:
   ```
   https://<your-railway-domain>/glasses
   ```
4. On your phone/laptop, open `https://<your-railway-domain>/speak`, tap **Start**, grant mic permission, and speak — the glasses sign in real time.

> Requirements (Meta docs): glasses software **v125+**, Meta AI app **v272+**.
> You can preview `/glasses` in a desktop browser sized to 600×600 before loading it on-device.

## Test the Python server independently

```bash
curl -X POST http://localhost:8000/pose \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello nice to meet you"}' \
  --output test.pose
ls -la test.pose  # Should be a non-zero binary file
```

## Build a two-handed ASL lexicon (WLASL → MediaPipe)

`python/build_lexicon.py` turns the [WLASL](https://github.com/dxli94/WLASL)
sign-language video dataset into a `.pose` lexicon that drops straight into the
pipeline above. For each English word it downloads a clip, crops it to the
sign's frame range, runs **MediaPipe Holistic** to extract body + face + both
hands, and saves a `.pose` in the exact format the fingerspelling lexicon uses
(so real signs and fingerspelled fallbacks concatenate cleanly).

> Academic / non-commercial use only — WLASL is released under a research
> license. This is intended for the hackathon demo.

### One-time builder setup

The builder needs **Python 3.12** (MediaPipe's legacy Holistic API isn't in the
3.13 wheels). `uv` will fetch 3.12 for you:

```bash
cd python
pip install uv
uv venv .venv-b312 --python 3.12
uv pip install --python .venv-b312/bin/python -r requirements-builder.txt

# Get the WLASL metadata (~12 MB)
mkdir -p wlasl
curl -L -o wlasl/WLASL_v0.3.json \
  https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json
```

### Build the lexicon

```bash
# A focused demo vocabulary (fast):
.venv-b312/bin/python build_lexicon.py --wlasl-json wlasl/WLASL_v0.3.json \
  --glosses book,drink,computer,help,family,learn,want,more,finish,name

# …or the 100 most-recorded glosses (slower; many clips, some dead links):
.venv-b312/bin/python build_lexicon.py --wlasl-json wlasl/WLASL_v0.3.json \
  --num-glosses 100
```

This writes `python/lexicon_wlasl/ase/<word>.pose` and
`python/lexicon_wlasl/index.csv`. Restart the Python server — it now signs those
words with real two-handed signs and fingerspells everything else.

If a build finds no usable clips, it will leave no `index.csv` behind. That is
intentional: an empty index makes `spoken-to-signed` reject `en/ase`, while no
index lets the app safely fall back to the bundled fingerspelling lexicon.

Useful flags: `--max-candidates N` (clips to try per word),
`--min-hand-fraction` (quality gate), `--use-bbox` (crop to the signer),
`--videos-dir DIR` + `--no-download` (reuse pre-downloaded videos),
`--overwrite` (rebuild existing words).

Verify a built lexicon (real sign + fingerspelled fallback in one phrase):

```bash
.venv/bin/text_to_gloss_to_pose --text "book computer zxq" --glosser simple \
  --spoken-language en --signed-language ase --lexicon lexicon_wlasl --pose /tmp/mix.pose
```
