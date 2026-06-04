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
3. Audio chunks are streamed to the Node server over a room WebSocket at `/ws/audio/<room-id>`
4. The server sends pose chunk metadata plus binary `.pose` clips to display clients in that same room
5. Each clip is fed to the `<pose-viewer>` web component as soon as the tiny playback buffer is ready
6. Chunks are queued so the avatar signs them in stream order
7. Transcript text (from Deepgram) is shown above the avatar in real time

Hackathon/demo fallback: the panel also renders curated no-mic phrase buttons
from `/api/demo/phrases`. Clicking a phrase posts to `/api/demo/pose`, streams
pose chunks via the Python server, and broadcasts transcript + pose events to
all display clients in the same room, including paired glasses.

### Sign → Speak Panel

1. User clicks "Start Camera" → browser opens webcam at 640×480
2. MediaPipe Hands (loaded from CDN) processes each video frame
3. Hand landmarks are drawn mirrored (selfie orientation) on a canvas
4. A **velocity-based segmentation** algorithm watches wrist movement:
   - Wrist velocity > 0.012 (normalized) = "signing active"
   - 6 pre-roll frames are included so the setup handshape is not lost
   - After 20 consecutive still frames → sign is complete, buffer is flushed
   - Final still frames are kept so end holds remain part of the sign
   - Buffer is force-flushed after 120 frames (~4 seconds)
5. The buffered frames (each as `{left_hand, right_hand}` 21×3 arrays) are POSTed to `/api/recognize`
6. If confidence ≥ 0.45, the recognized gloss is shown as a removable word chip
   with confidence when available
7. Users can remove a chip, undo the removal, or choose a better top-5 candidate
   from the recognizer before confirming the phrase
8. After 2.5 seconds of silence (no new glosses), or when the user clicks "Use Phrase",
   the accumulated ASL gloss is:
   - Translated to natural English via `/api/gloss-to-english` (OpenAI)
   - Spoken aloud via `/api/tts` (ElevenLabs TTS)

---

## Layer 2 — Node.js Server (`server.js`)

Built with Express + the `ws` WebSocket library. Runs on port 3000.

### Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `DEEPGRAM_API_KEY` | Required for Speak→Sign live speech-to-text. |
| `OPENAI_API_KEY` | Optional. English → ASL gloss. Falls back to raw transcript. |
| `OPENAI_MODEL` | Default `gpt-4o-mini` (fast/cheap). |
| `ELEVENLABS_API_KEY` | Optional. TTS for Sign→Speak mode. |
| `ELEVENLABS_VOICE_ID` | Default Rachel voice. |
| `POSE_SERVER_URL` | Default `http://localhost:8000`. Python server URL. |
| `GLOSS_TIMEOUT_MS` | Default 1500ms. OpenAI timeout; falls back gracefully. |
| `STREAM_CHUNK_TARGET_WORDS` | Default 3. Normal finalized-word chunk size for live signing. |
| `STREAM_CHUNK_MAX_WORDS` | Default 5. Largest forced chunk at speech end/disconnect. |
| `STREAM_CHUNK_IDLE_MS` | Default 80ms. Flush delay for a short leftover finalized chunk. |
| `STREAM_OPENAI_GLOSS` | Default off. Set `1` to use OpenAI for streaming chunk glossing. |

### Session Rooms and WebSocket Connection Types

Each conversation is isolated in an in-memory room keyed by a URL-safe room ID.
Opening `/` or `/speak` without a room redirects to `/speak?room=<room-id>`.
Glasses and companion devices join the same conversation by using the same room
query parameter.

Each room maintains two WebSocket roles:

- **Audio clients** (`/ws/audio/<room-id>`): Browsers streaming microphone PCM data
- **Display clients** (`/ws/display/<room-id>`): Browsers waiting to receive `.pose` blobs and transcript JSON

During migration, `/audio?room=<room-id>` and `/?room=<room-id>` remain accepted
as compatibility shims. WebSocket clients without a valid room ID are rejected.

When a browser connects via `/ws/audio/<room-id>`:
1. A **Deepgram live transcription session** is opened (`nova-3` model, 16 kHz mono PCM)
2. Audio chunks from the browser are forwarded directly to Deepgram
3. Deepgram fires `Transcript` events with interim + final results
4. Interim and final transcript messages are sent only to display clients in the same room
5. **Only finalized words** feed the pose pipeline (interim words may change)
6. Finalized words are chunked into small signable units, normally 1-3 words
7. Each chunk is incrementally glossed, posed by the Python server, and sent as soon as it is ready
8. Display clients receive JSON pose metadata followed by the binary `.pose` clip
9. The browser keeps a tiny playback buffer and progressively renders clips in order

### Speak → Sign Pipeline (streaming)

```
Deepgram final transcript chunk
  → expandDigitsForSigning()   (e.g. "3" → "three")
  → incrementalEnglishToAslGlossResult()
       default: local low-latency gloss
       optional: OpenAI GPT with STREAM_OPENAI_GLOSS=1
  → generatePose()             (Python /pose endpoint → .pose binary clip)
  → broadcast pose metadata
  → broadcast binary clip
  → display playback buffer
  → pose-viewer render
```

An LRU gloss cache (default 128 entries) avoids redundant OpenAI calls for repeated phrases.

Pose generation is **serialized per audio connection** via a promise chain so
chunks from one speaker broadcast in order. Multiple rooms can run
simultaneously. Multiple audio clients in the same room are allowed; their
chunks may interleave at chunk boundaries.

### ASL Gloss Prompt (sent to OpenAI)

> You are an expert ASL interpreter. Convert English into ASL gloss. Rules: use topic-comment word order, time first; drop articles and forms of "to be"; use uninflected base words (no -ing/-ed/plural -s); keep WH-question words in natural ASL position; expand contractions. Output ONLY gloss as space-separated UPPERCASE words, no punctuation.

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirects to a fresh `/speak?room=<room-id>` session |
| `GET` | `/speak?room=<id>` | Serves `index.html` for the requested room |
| `GET` | `/glasses?room=<id>` | Serves `glasses.html` for the requested room |
| `GET` | `/api/sessions/new` | Returns a fresh room ID and matching client URLs |
| `GET` | `/api/sessions/:roomId/pairing` | Returns the tokenized glasses pairing link and QR SVG URL |
| `GET` | `/api/sessions/:roomId/glasses-qr.svg` | Renders the QR code for the tokenized glasses link |
| `GET` | `/api/health` | Reports configured services and pose-server reachability |
| `GET` | `/api/demo/phrases` | Returns the curated no-mic phrase list |
| `POST` | `/api/demo/pose` | Generates a demo pose and broadcasts it to a private room |
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
- Extracts a **126-dimensional hand-shape feature vector** per frame:
  - Normalizes each hand by centering at wrist (landmark 0) and dividing by max finger distance
  - Concatenates left hand (63 dims) + right hand (63 dims)
- Builds a compact temporal template by resampling the sign to 24 frames
- Adds normalized left/right wrist trajectory and hand visibility signals
- Runs a fast cosine-similarity pass against all lexicon means to shortlist candidates
- Reranks the top candidates with banded DTW over the temporal templates
- Returns the winning gloss plus optional top-5 reranked candidates:
  `{"gloss": "word", "confidence": 0.xx, "mean_confidence": 0.xx, "temporal_confidence": 0.xx, "candidates": [{"gloss": "word", "confidence": 0.xx, "mean_confidence": 0.xx, "temporal_confidence": 0.xx}]}`
- The browser also accepts future-compatible candidate field names such as
  `alternatives`, `top_candidates`, or `top5`, and candidate labels named
  `gloss`, `word`, `label`, `text`, or `value`.

#### `GET /health`
- Returns `{"status": "ok"}`

### Sign Recognition Lexicon Loading

At startup, the server reads every `.pose` file listed in `lexicon_wlasl/index.csv`. For each:
1. Reads the `.pose` binary using the `pose_format` library
2. Extracts `LEFT_HAND_LANDMARKS` and `RIGHT_HAND_LANDMARKS` components
3. Skips frames where no hand is visible above the confidence threshold
4. Computes a mean 126-dim hand-shape vector for coarse retrieval
5. Computes a fixed-length temporal template with wrist trajectory for reranking
6. Stores both references keyed by gloss

This keeps inference practical: O(N) dot products for the coarse pass, then
temporal matching only for the top-K candidates.

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
  → PCM audio chunks (WebSocket /ws/audio/<room-id>)
  → Deepgram nova-3 (cloud STT)
  → Final transcript chunks
  → expandDigitsForSigning() (e.g. "3" → "three")
  → incremental local gloss (OpenAI optional via STREAM_OPENAI_GLOSS=1)
  → Python /pose (cached lexicon lookup + postprocessing)
  → pose metadata + .pose binary clip (WebSocket /ws/display/<room-id>)
  → browser playback buffer
  → <pose-viewer> web component (progressive skeleton avatar animation)
```

### Speak → Sign (demo path)

```
Curated phrase button
  → POST /api/demo/pose with roomId + phraseId
  → streaming text chunks
  → incremental gloss
  → Python /pose per chunk
  → transcript/gloss/pose metadata + .pose clips broadcast to /ws/display/<room-id>
```

### Sign → Speak (full path)

```
Webcam (browser)
  → MediaPipe Hands (client-side, CDN)
  → 21 hand landmarks per frame (x,y,z normalized)
  → Velocity-based segmentation + pre-roll/final-hold capture
  → POST /api/recognize → Python /recognize
  → Cosine shortlist + temporal template rerank vs. WLASL lexicon
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
