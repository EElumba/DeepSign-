# DeepSign — Project Overview

## What This Project Does

DeepSign is a real-time, bidirectional American Sign Language (ASL) translation web application. It has two operating modes that work in opposite directions:

1. **Speak → Sign**: You speak into a microphone; the app transcribes your speech, translates it to ASL grammar, and plays an animated avatar that signs the words.
2. **Sign → Speak**: You sign in front of your camera; the app recognizes your hand gestures and converts them to spoken English audio.

The intended use case is accessibility tooling — for example, allowing a Deaf person wearing Meta smart glasses to communicate with hearing people in real time, or allowing a hearing person to sign to a Deaf person.

---

## System Architecture

The app is split into three layers:

```
Browser (index.html)
  ↕ WebSocket / REST
Node.js Server (server.js, port 3000)
  ↕ HTTP / subprocess
Python FastAPI Server (python/server.py, port 8000)
```

All three must be running simultaneously.

---

## Layer 1 — Browser Frontend (`index.html`)

A single-page HTML/JS app. No framework, no build step. The UI has:

- A mode toggle bar: "Speak to Sign" / "Sign to Speak"
- A live connection status dot (green = live, red = error)

### Speak → Sign Panel

1. User clicks "Start" → browser requests mic permission
2. Raw 16 kHz PCM audio is captured via `AudioContext.createScriptProcessor`
3. Audio chunks are streamed to the Node server over a WebSocket at `/audio`
4. The server sends back binary `.pose` blobs (one per signed phrase)
5. Each blob is fed to the `<pose-viewer>` web component, which animates a skeleton avatar signing the phrase
6. Phrases are queued so the avatar signs them in order
7. Transcript text (from Deepgram) is shown above the avatar in real time

### Sign → Speak Panel

1. User clicks "Start Camera" → browser opens webcam at 640×480
2. MediaPipe Hands (loaded from CDN) processes each video frame
3. Hand landmarks are drawn mirrored (selfie orientation) on a canvas
4. A **velocity-based segmentation** algorithm watches wrist movement:
   - Wrist velocity > 0.012 (normalized) = "signing active"
   - After 20 consecutive still frames → sign is complete, buffer is flushed
   - Buffer is force-flushed after 120 frames (~4 seconds)
5. The buffered frames (each as `{left_hand, right_hand}` 21×3 arrays) are POSTed to `/api/recognize`
6. If confidence ≥ 0.45, the recognized gloss word is appended to a running list
7. After 2.5 seconds of silence (no new glosses), the accumulated ASL gloss is:
   - Translated to natural English via `/api/gloss-to-english` (OpenAI)
   - Spoken aloud via `/api/tts` (ElevenLabs TTS)

---

## Layer 2 — Node.js Server (`server.js`)

Built with Express + the `ws` WebSocket library. Runs on port 3000.

### Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `DEEPGRAM_API_KEY` | Required. Live speech-to-text. |
| `OPENAI_API_KEY` | Optional. English → ASL gloss. Falls back to raw transcript. |
| `OPENAI_MODEL` | Default `gpt-4o-mini` (fast/cheap). |
| `ELEVENLABS_API_KEY` | Optional. TTS for Sign→Speak mode. |
| `ELEVENLABS_VOICE_ID` | Default Rachel voice. |
| `POSE_SERVER_URL` | Default `http://localhost:8000`. Python server URL. |
| `GLOSS_TIMEOUT_MS` | Default 1500ms. OpenAI timeout; falls back gracefully. |
| `FINAL_FLUSH_DELAY_MS` | Default 120ms. Safety flush delay for partial utterances. |

### WebSocket Connection Types

Two WebSocket connection pools are maintained:

- **Audio clients** (`/audio` URL): Browsers streaming microphone PCM data
- **Display clients** (root URL): Browsers waiting to receive `.pose` blobs and transcript JSON

When a browser connects via `/audio`:
1. A **Deepgram live transcription session** is opened (`nova-3` model, 16 kHz mono PCM)
2. Audio chunks from the browser are forwarded directly to Deepgram
3. Deepgram fires `Transcript` events with interim + final results
4. **Only finalized words** feed the pose pipeline (interim words may change)
5. Words are buffered up to 16 words or until Deepgram signals `speech_final`
6. The buffered phrase is **glossed to ASL** via OpenAI GPT then **posed** by the Python server

### Speak → Sign Pipeline (per utterance)

```
Deepgram final transcript
  → expandDigitsForSigning()   (e.g. "3" → "three")
  → englishToAslGloss()        (OpenAI GPT: English → ASL gloss grammar)
  → generatePose()             (Python /pose endpoint → .pose binary)
  → broadcastToDisplays()      (sends binary to all display WebSocket clients)
```

An LRU gloss cache (default 128 entries) avoids redundant OpenAI calls for repeated phrases.

Pose generation is **serialized per connection** via a promise chain so clips broadcast in order.

### ASL Gloss Prompt (sent to OpenAI)

> You are an expert ASL interpreter. Convert English into ASL gloss. Rules: use topic-comment word order, time first; drop articles and forms of "to be"; use uninflected base words (no -ing/-ed/plural -s); keep WH-question words in natural ASL position; expand contractions. Output ONLY gloss as space-separated UPPERCASE words, no punctuation.

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/api/recognize` | Proxies landmark frames to Python `/recognize` |
| `POST` | `/api/gloss-to-english` | Calls OpenAI to convert ASL gloss → natural English |
| `POST` | `/api/tts` | Proxies text to ElevenLabs TTS; streams MP3 audio back |

---

## Layer 3 — Python FastAPI Server (`python/server.py`)

Runs on port 8000. Requires the Python virtual environment in `python/.venv`.

### Key Endpoints

#### `POST /pose`
- Accepts `{"text": "..."}` (the ASL gloss string)
- Runs the `text_to_gloss_to_pose` CLI (from the `spoken-to-signed` package)
- Returns raw `.pose` binary (application/octet-stream)
- Results are LRU-cached (default 128 entries)

After the CLI generates a pose, a **postprocessing pipeline** runs:

1. **Body size stabilization**: Scales shoulder-to-hip body height to a target of 210 pixels so avatars from different source videos appear the same size. Clamps scale factor to [0.65, 1.35].
2. **Face stabilization**: Normalizes face landmark bounding box to target width/height (116×120px) and repositions the face relative to shoulder center. Prevents the head from growing/shrinking between signs.
3. **Y offset**: Shifts all image-space landmarks up by 45 pixels (avatar centering).
4. **Playback speed**: Multiplies the stored FPS by 1.6 (default) so the avatar signs at 160% native speed.

#### `POST /recognize`
- Accepts `{"frames": [{left_hand: [[x,y,z]×21]|null, right_hand: ...}, ...]}`
- Extracts a **126-dimensional feature vector** per frame:
  - Normalizes each hand by centering at wrist (landmark 0) and dividing by max finger distance
  - Concatenates left hand (63 dims) + right hand (63 dims)
- Averages across all frames, L2-normalizes the result
- Computes cosine similarity against every sign in the loaded lexicon
- Returns `{"gloss": "word", "confidence": 0.xx}`

#### `GET /health`
- Returns `{"status": "ok"}`

### Sign Recognition Lexicon Loading

At startup, the server reads every `.pose` file listed in `lexicon_wlasl/index.csv`. For each:
1. Reads the `.pose` binary using the `pose_format` library
2. Extracts `LEFT_HAND_LANDMARKS` and `RIGHT_HAND_LANDMARKS` components
3. Computes a mean 126-dim feature vector across all frames with hand confidence > 0.1
4. L2-normalizes and stores as a reference vector keyed by gloss

This gives O(N) recognition at inference (dot product against all N signs).

### Lexicon Resolution Priority

```
1. LEXICON_DIR env var (explicit override)
2. python/lexicon_wlasl/ if index.csv exists and has valid .pose files (real signs)
3. Bundled fingerspelling lexicon from spoken-to-signed package (letter-by-letter fallback)
```

The CLI always falls back to fingerspelling for out-of-vocabulary words regardless of lexicon.

---

## Lexicon Data (`python/lexicon_wlasl/`)

Contains **2,000 ASL gloss `.pose` files** covering common vocabulary:

- Family, food, school/work, health, time, questions, conversation-repair
- Each file: `ase/<gloss>.pose` (e.g. `ase/hello.pose`, `ase/family.pose`)
- `index.csv`: maps each gloss to its pose file path and language metadata

Source: **WLASL (Word-Level American Sign Language)** dataset  
License: C-UDA (academic/computational use only, no commercial redistribution)

---

## Lexicon Builder Scripts

### `python/build_lexicon.py` — Build from Raw Video

Uses **MediaPipe Holistic** (legacy, requires Python 3.12 + separate venv) to extract landmarks from WLASL videos.

**Pipeline per gloss:**
1. Load WLASL JSON metadata
2. Download video via `yt-dlp` (or use pre-downloaded)
3. Crop to the sign's frame range (and optionally the signer bounding box)
4. Run MediaPipe Holistic → 543+ body/face/hand landmarks per frame
5. Check hand detection fraction ≥ 30% of frames (quality gate)
6. Save as `.pose` file, append to `index.csv`

**Example usage:**
```bash
.venv-b312/bin/python build_lexicon.py \
  --wlasl-json wlasl/WLASL_v0.3.json \
  --glosses hello,family,help,book
```

### `python/import_processed_wlasl.py` — Import Pre-processed Landmarks

Faster alternative: converts the Kaggle "MuteMotion: WLASL MediaPipe Encoded" archive (pre-extracted `.npz` landmarks) directly into `.pose` files — no video download or MediaPipe needed.

**Key steps:**
1. Load `landmarks_V3.npz` (shape: N samples × 553 points × 3 coords)
2. Reorder from Kaggle V3 format `[Right Hand, Left Hand, Pose, Face]` to pose-format order `[Pose, Face, Left Hand, Right Hand]`
3. Scale normalized [0–1] coordinates to pixel space (default 640×480)
4. Apply **Savitzky-Golay smoothing** (window=5, polyorder=2) per landmark
5. Quality scoring: penalizes jitter, wrist jumps, low hand visibility, shoulder instability
6. Pick the best candidate clip per gloss
7. Write `.pose` and update `index.csv`

**Example usage:**
```bash
python import_processed_wlasl.py \
  --source-dir wlasl_processed_source/archive \
  --num-glosses 500
```

### `python/normalize_pose_dimensions.py` — Normalize Coordinate Space

Since WLASL clips come from different source videos (different frame sizes), this script rescales all `.pose` files to a uniform canvas (default 480×320) while preserving aspect ratio via letterboxing.

---

## Data Flow Summary

### Speak → Sign (full path)

```
Microphone (browser)
  → PCM audio chunks (WebSocket /audio)
  → Deepgram nova-3 (cloud STT)
  → Final transcript text
  → expandDigitsForSigning() (e.g. "3" → "three")
  → OpenAI GPT-4o-mini (English → ASL gloss)
  → Python /pose (text_to_gloss_to_pose CLI + postprocessing)
  → .pose binary (WebSocket broadcast)
  → <pose-viewer> web component (skeleton avatar animation)
```

### Sign → Speak (full path)

```
Webcam (browser)
  → MediaPipe Hands (client-side, CDN)
  → 21 hand landmarks per frame (x,y,z normalized)
  → Velocity-based segmentation (wrist movement threshold)
  → POST /api/recognize → Python /recognize
  → Cosine similarity vs. WLASL lexicon feature vectors
  → Recognized gloss word
  → Accumulate glosses until 2.5s idle
  → POST /api/gloss-to-english → OpenAI (ASL gloss → English)
  → POST /api/tts → ElevenLabs (English → MP3 audio)
  → Audio playback in browser
```

---

## Key External Libraries & Services

| Tool | Role |
|---|---|
| Deepgram (nova-3) | Cloud live speech-to-text |
| OpenAI GPT-4o-mini | English ↔ ASL gloss translation |
| ElevenLabs | Text-to-speech (Sign→Speak mode) |
| `spoken-to-signed` | Python CLI: text → ASL pose animation |
| `pose_format` | Python library: read/write/manipulate .pose files |
| `pose-viewer` | Web component: render .pose skeleton animation |
| MediaPipe Hands | Client-side hand landmark detection |
| MediaPipe Holistic | Server-side full-body landmark extraction (lexicon builder only) |
| Express + ws | Node.js HTTP + WebSocket server |
| FastAPI + uvicorn | Python HTTP server |
| WLASL dataset | Source of real ASL signing videos / landmarks |

---

## How to Run

### 1. Python server
```bash
cd python
uvicorn server:app --port 8000
```

### 2. Node server
```bash
# At project root (copy .env.example to .env and fill in API keys first)
npm start
```

### 3. Open browser
Navigate to `http://localhost:3000`

---

## File Structure

```
DeepSign-/
├── index.html                    # Single-page frontend (Speak↔Sign UI)
├── server.js                     # Node.js WebSocket + REST server
├── package.json                  # Node dependencies
├── .env.example                  # API key template
└── python/
    ├── server.py                 # FastAPI: /pose, /recognize, /health
    ├── build_lexicon.py          # Build lexicon from raw WLASL videos
    ├── import_processed_wlasl.py # Build lexicon from Kaggle .npz landmarks
    ├── normalize_pose_dimensions.py  # Rescale .pose files to uniform size
    └── lexicon_wlasl/
        ├── README.md
        ├── index.csv             # Gloss → pose file mapping
        └── ase/
            ├── hello.pose
            ├── family.pose
            └── ...               # 2,000 .pose files total
```
