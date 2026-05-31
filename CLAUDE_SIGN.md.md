# AccessLink — Sign-to-Voice Pipeline (Reverse Mode)
> Paste this file into your AI coding assistant alongside CLAUDE.md for full context.
> This file covers Option B: MediaPipe Hands → WLASL TGCN classifier → sentence builder → ElevenLabs TTS
> Last updated: May 2026

---

## What this module does

This is the **reverse pipeline** — the second mode of AccessLink. A mute user signs into the camera. MediaPipe extracts hand landmarks in real time. A pretrained TGCN model (trained on WLASL) classifies each sign into a word. A word buffer accumulates signs into a sentence. The Claude API converts the gloss sequence into natural English. ElevenLabs speaks it aloud.

**User experience:** The mute person signs "HELLO NAME ALEX NICE MEET". The app speaks: *"Hello, my name is Alex. Nice to meet you."*

---

## How it fits into the existing app

The existing AccessLink app runs a **voice-to-sign** forward pipeline (Deepgram → sign animation). This reverse pipeline runs as a **second mode** toggled by a button in the UI. Both modes share:
- The same camera feed (`AROverlay.js`)
- The same Snowflake logger (`SnowflakeLogger.js`)
- The same config (`config.js`)

New mode-specific components live in `modules/reverse/`.

---

## Full Pipeline

```
[Camera feed]
     │  getUserMedia({video:true}) — already open from AROverlay
     ▼
[Stage R1: Hand Detection]
     │  MediaPipe Hands — 21 landmarks × (x,y,z) per hand × 2 hands
     ▼
[Stage R2: Frame Buffer]
     │  Accumulate N_FRAMES (30) of landmark sequences
     │  Sliding window — append each frame, drop oldest when full
     ▼
[Stage R3: WLASL TGCN Classifier]
     │  Input:  (30 frames × 55 keypoints) tensor
     │  Output: { gloss: "hello", confidence: 0.94 }
     │  Threshold: only accept predictions above CONFIDENCE_THRESHOLD (0.75)
     ▼
[Stage R4: Gloss Buffer]
     │  Accumulate confirmed gloss words
     │  Flush trigger: 1.5s pause in signing (no new confident prediction)
     ▼
[Stage R5: Sentence Builder — Claude API]
     │  POST to Anthropic API with gloss sequence
     │  Returns natural English sentence
     ▼
[Stage R6: TTS — ElevenLabs]
     │  POST sentence text → ElevenLabs Flash v2.5
     │  Stream audio → play through speaker
     ▼
[Hearing person hears: natural spoken English]

                         └──── async ────→ [Snowflake: reverse_events log]
```

---

## New File Structure (additions to existing accesslink/)

```
accesslink/
├── [all existing files unchanged]
│
├── modules/
│   ├── [all existing modules unchanged]
│   │
│   └── reverse/
│       ├── HandDetector.js        # MediaPipe Hands wrapper — emits landmark frames
│       ├── FrameBuffer.js         # Sliding window of N_FRAMES landmark sequences
│       ├── SignClassifier.js      # WLASL TGCN model wrapper — frame buffer → gloss
│       ├── GlossBuffer.js         # Accumulates confirmed glosses, detects signing pause
│       ├── SentenceBuilder.js     # Claude API — gloss sequence → natural English
│       └── TTSPlayer.js           # ElevenLabs TTS → audio output
│
├── python/
│   ├── server.py                  # EXISTING — FastAPI pose server (forward pipeline)
│   ├── sign_classifier.py         # NEW — TGCN inference endpoint
│   ├── models/
│   │   └── tgcn_wlasl100/         # NEW — pretrained TGCN weights from Hugging Face
│   │       ├── pytorch_model.bin
│   │       └── config.ini
│   └── requirements_reverse.txt   # NEW — mediapipe, torch, huggingface_hub
│
└── config.js                      # Add reverse pipeline keys here
```

---

## Configuration additions (add to existing config.js)

```js
// Add these to the existing CONFIG object in config.js

// Reverse pipeline — sign to voice
REVERSE_MODE_ENABLED:       false,          // toggled by UI button
ELEVENLABS_API_KEY:         '',             // get from elevenlabs.io
ELEVENLABS_VOICE_ID:        'EXAVITQu4vr4xnSDxMaL',  // default: Bella (natural female)
ELEVENLABS_MODEL_ID:        'eleven_flash_v2_5',       // lowest latency model

ANTHROPIC_API_KEY:          '',             // for sentence builder
CLAUDE_MODEL:               'claude-sonnet-4-20250514',

// Classifier settings
WLASL_VOCAB_SIZE:           100,            // 100 | 300 | 1000 | 2000
CONFIDENCE_THRESHOLD:       0.75,           // minimum confidence to accept a prediction
N_FRAMES:                   30,             // frames per classification window (~1 sec at 30fps)
SIGNING_PAUSE_MS:           1500,           // ms of no new sign → flush gloss buffer
CLASSIFIER_ENDPOINT:        'http://localhost:8001/classify',  // Python FastAPI

// TTS settings
TTS_AUTOPLAY:               true,
TTS_VOLUME:                 1.0,
```

---

## Python Setup — New Classifier Server

This runs as a **second Python server** on port 8001, separate from the existing pose server on port 8000.

### Install (run once)

```bash
cd python

# Create a new venv for the classifier (separate from existing .venv)
python3 -m venv .venv-reverse
.venv-reverse/bin/pip install -r requirements_reverse.txt
```

### requirements_reverse.txt

```
fastapi==0.115.0
uvicorn==0.30.0
mediapipe==0.10.14
torch==2.3.0
numpy==1.26.4
huggingface_hub==0.24.0
python-multipart==0.0.9
```

### Download pretrained WLASL TGCN weights

```bash
# Run once — downloads ~50MB of pretrained weights
cd python
.venv-reverse/bin/python - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sharonn18/tgcn-wlasl",
    local_dir="models/tgcn_wlasl100",
    allow_patterns=["asl100/*", "configs/asl100.ini", "tgcn_model.py", "configs.py"]
)
print("Downloaded TGCN weights for WLASL100")
EOF
```

### sign_classifier.py — FastAPI classifier server

```python
# python/sign_classifier.py
# Run: .venv-reverse/bin/uvicorn sign_classifier:app --port 8001

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'tgcn_wlasl100'))

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from tgcn_model import TGCN
from configs import load_config

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model once on startup ──────────────────────────────────────────────
cfg = load_config('models/tgcn_wlasl100/configs/asl100.ini')
model = TGCN(cfg)
weights = torch.load(
    'models/tgcn_wlasl100/asl100/pytorch_model.bin',
    map_location='cpu'
)
model.load_state_dict(weights)
model.eval()

# WLASL100 gloss labels — top 100 most common ASL words
# Full list from WLASL repo: github.com/dxli94/WLASL
WLASL100_LABELS = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who",
    "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year",
    "yes", "all", "black", "cool", "finish", "hot", "like", "many", "mother",
    "now", "orange", "table", "thanksgiving", "what", "white", "wrong",
    "accident", "apple", "bird", "change", "color", "corn", "cow", "dance",
    "dark", "day", "doctor", "dog", "eat", "every", "family", "fast", "fish",
    "forget", "give", "glass", "good", "gray", "green", "happy", "hat",
    "hearing", "horse", "kiss", "language", "last", "letter", "man", "money",
    "month", "more", "name", "need", "nurse", "old", "pay", "pizza", "play",
    "purple", "right", "school", "secretary", "short", "shower", "son",
    "sorry", "spend", "spring", "store", "student", "tall", "tell", "thursday",
    "time", "uncle", "want", "water", "woman", "work", "world", "write"
]

# ── Request model ────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    # frames: list of 30 frames, each frame = 55 keypoints flattened to 165 floats
    # (body: 33 × 3) + (left hand: 21 × 3) + (right hand: 21 × 3) - mapped to 55 used
    frames: List[List[float]]   # shape: (30, 165)

@app.post("/classify")
async def classify(req: ClassifyRequest):
    if len(req.frames) < 10:
        return {"gloss": None, "confidence": 0.0, "error": "not enough frames"}

    # Pad or trim to exactly 30 frames
    frames = req.frames[-30:] if len(req.frames) > 30 else req.frames
    while len(frames) < 30:
        frames.append(frames[-1])  # repeat last frame to pad

    x = torch.tensor([frames], dtype=torch.float32)  # (1, 30, 165)

    with torch.no_grad():
        logits = model(x)                             # (1, num_classes)
        probs  = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

    gloss = WLASL100_LABELS[pred_idx.item()]
    conf  = round(confidence.item(), 4)

    return {
        "gloss":      gloss,
        "confidence": conf,
        "top3": [
            {"gloss": WLASL100_LABELS[i], "confidence": round(probs[0][i].item(), 3)}
            for i in probs[0].topk(3).indices.tolist()
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "vocab_size": len(WLASL100_LABELS)}
```

---

## JavaScript Modules — Reverse Pipeline

### `modules/reverse/HandDetector.js`

```js
// HandDetector.js
// Wraps MediaPipe Hands. Runs on each camera frame. Emits landmark arrays.

import { Hands } from '@mediapipe/hands';

export class HandDetector {
  constructor(videoElement) {
    this.video = videoElement;
    this.onLandmarks = null; // callback: (landmarks: Float32Array) => void

    this.hands = new Hands({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });

    this.hands.setOptions({
      maxNumHands:          2,
      modelComplexity:      1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence:  0.7,
    });

    this.hands.onResults(results => this._onResults(results));
  }

  start() {
    // Call this in the rAF loop or setInterval at ~30fps
    this._rafId = requestAnimationFrame(this._loop.bind(this));
  }

  stop() {
    cancelAnimationFrame(this._rafId);
  }

  async _loop() {
    await this.hands.send({ image: this.video });
    this._rafId = requestAnimationFrame(this._loop.bind(this));
  }

  _onResults(results) {
    // Flatten all landmarks into a single Float32Array
    // Format: [lh_x0, lh_y0, lh_z0, lh_x1 ... rh_x0, rh_y0, rh_z0 ...]
    // If hand not visible, fill with zeros
    const flat = new Float32Array(21 * 3 * 2); // 126 floats: both hands

    const lh = results.multiHandLandmarks?.[0];
    const rh = results.multiHandLandmarks?.[1];

    if (lh) lh.forEach((p, i) => {
      flat[i * 3]     = p.x;
      flat[i * 3 + 1] = p.y;
      flat[i * 3 + 2] = p.z;
    });

    if (rh) rh.forEach((p, i) => {
      flat[63 + i * 3]     = p.x;
      flat[63 + i * 3 + 1] = p.y;
      flat[63 + i * 3 + 2] = p.z;
    });

    if (this.onLandmarks) this.onLandmarks(flat);
  }
}
```

### `modules/reverse/FrameBuffer.js`

```js
// FrameBuffer.js
// Sliding window of the last N frames of landmark data.

export class FrameBuffer {
  constructor(nFrames = 30) {
    this.nFrames = nFrames;
    this.buffer  = [];   // array of Float32Array (126 floats each)
  }

  push(landmarks) {
    this.buffer.push(Array.from(landmarks));
    if (this.buffer.length > this.nFrames) this.buffer.shift();
  }

  isFull() {
    return this.buffer.length >= this.nFrames;
  }

  getFrames() {
    // Returns array of arrays — shape (N, 126)
    // Pad shorter landmark arrays to 165 to match TGCN input
    return this.buffer.map(frame => {
      const padded = new Array(165).fill(0);
      frame.forEach((v, i) => { if (i < 165) padded[i] = v; });
      return padded;
    });
  }

  clear() {
    this.buffer = [];
  }
}
```

### `modules/reverse/SignClassifier.js`

```js
// SignClassifier.js
// Sends frame buffer to Python TGCN server. Returns gloss prediction.

export class SignClassifier {
  constructor(endpoint, confidenceThreshold) {
    this.endpoint  = endpoint;            // CONFIG.CLASSIFIER_ENDPOINT
    this.threshold = confidenceThreshold; // CONFIG.CONFIDENCE_THRESHOLD
    this._busy     = false;
  }

  async classify(frames) {
    if (this._busy) return null;  // drop if previous call still pending
    this._busy = true;

    try {
      const res = await fetch(this.endpoint, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ frames }),
      });
      const data = await res.json();

      if (data.confidence >= this.threshold) {
        return { gloss: data.gloss, confidence: data.confidence };
      }
      return null; // below threshold — discard

    } catch (e) {
      if (CONFIG.DEBUG_MODE) console.warn('Classifier error:', e.message);
      return null;
    } finally {
      this._busy = false;
    }
  }
}
```

### `modules/reverse/GlossBuffer.js`

```js
// GlossBuffer.js
// Accumulates confirmed gloss words. Detects end-of-sentence pause. Flushes to SentenceBuilder.

export class GlossBuffer {
  constructor(pauseMs, onSentenceReady) {
    this.pauseMs         = pauseMs;        // CONFIG.SIGNING_PAUSE_MS
    this.onSentenceReady = onSentenceReady;
    this.glosses         = [];
    this._lastSignTime   = null;
    this._flushTimer     = null;
    this._lastGloss      = null;          // dedupe: skip same gloss twice in a row
  }

  push(gloss) {
    // Deduplicate consecutive identical predictions
    if (gloss === this._lastGloss) return;
    this._lastGloss  = gloss;
    this._lastSignTime = Date.now();

    this.glosses.push(gloss);

    // Reset flush timer on every new sign
    clearTimeout(this._flushTimer);
    this._flushTimer = setTimeout(() => this._flush(), this.pauseMs);

    if (CONFIG.DEBUG_MODE) console.log('Gloss buffer:', this.glosses);
  }

  _flush() {
    if (this.glosses.length === 0) return;
    const sentence = [...this.glosses];
    this.glosses    = [];
    this._lastGloss = null;
    this.onSentenceReady(sentence);
  }

  clear() {
    clearTimeout(this._flushTimer);
    this.glosses    = [];
    this._lastGloss = null;
  }
}
```

### `modules/reverse/SentenceBuilder.js`

```js
// SentenceBuilder.js
// Sends raw ASL gloss sequence to Claude API → returns natural English sentence.

export class SentenceBuilder {
  async build(glosses) {
    // glosses: ['hello', 'name', 'alex', 'nice', 'meet']
    // Returns: "Hello, my name is Alex. Nice to meet you."

    const prompt = `You are an ASL-to-English interpreter. Convert this sequence of ASL gloss words into a single natural, grammatically correct English sentence. Respond with ONLY the sentence — no explanation, no punctuation marks before the sentence, no quotes.

ASL glosses: ${glosses.join(' ')}

Natural English:`;

    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method:  'POST',
        headers: {
          'Content-Type':         'application/json',
          'x-api-key':            CONFIG.ANTHROPIC_API_KEY,
          'anthropic-version':    '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model:      CONFIG.CLAUDE_MODEL,
          max_tokens: 150,
          messages:   [{ role: 'user', content: prompt }],
        }),
      });

      const data = await res.json();
      return data.content?.[0]?.text?.trim() || glosses.join(' ');

    } catch (e) {
      console.warn('SentenceBuilder failed, using raw glosses:', e.message);
      return glosses.join(' ');  // fallback: speak raw gloss words
    }
  }
}
```

### `modules/reverse/TTSPlayer.js`

```js
// TTSPlayer.js
// Sends text to ElevenLabs Flash v2.5. Streams audio to speaker.
// Falls back to Web Speech API if ElevenLabs key is missing.

export class TTSPlayer {
  constructor() {
    this._playing = false;
  }

  async speak(text) {
    if (!text || this._playing) return;
    this._playing = true;

    try {
      if (CONFIG.ELEVENLABS_API_KEY) {
        await this._elevenLabs(text);
      } else {
        await this._webSpeech(text);
      }
    } finally {
      this._playing = false;
    }
  }

  async _elevenLabs(text) {
    const res = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_VOICE_ID}/stream`,
      {
        method:  'POST',
        headers: {
          'xi-api-key':    CONFIG.ELEVENLABS_API_KEY,
          'Content-Type':  'application/json',
        },
        body: JSON.stringify({
          text,
          model_id:         CONFIG.ELEVENLABS_MODEL_ID,
          voice_settings:   { stability: 0.5, similarity_boost: 0.75 },
        }),
      }
    );

    const arrayBuffer = await res.arrayBuffer();
    const audioCtx    = new AudioContext();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const source      = audioCtx.createBufferSource();
    source.buffer     = audioBuffer;
    source.connect(audioCtx.destination);
    source.start();

    await new Promise(resolve => { source.onended = resolve; });
  }

  _webSpeech(text) {
    return new Promise(resolve => {
      const utt      = new SpeechSynthesisUtterance(text);
      utt.lang       = 'en-US';
      utt.rate       = 1.0;
      utt.onend      = resolve;
      utt.onerror    = resolve;
      speechSynthesis.speak(utt);
    });
  }
}
```

---

## Wiring it all together — reverse mode in main.js

```js
// Add to main.js — reverse mode toggle

import { HandDetector }    from './modules/reverse/HandDetector.js';
import { FrameBuffer }     from './modules/reverse/FrameBuffer.js';
import { SignClassifier }  from './modules/reverse/SignClassifier.js';
import { GlossBuffer }     from './modules/reverse/GlossBuffer.js';
import { SentenceBuilder } from './modules/reverse/SentenceBuilder.js';
import { TTSPlayer }       from './modules/reverse/TTSPlayer.js';

let reverseActive = false;
let handDetector, frameBuffer, classifier, glossBuffer, sentenceBuilder, ttsPlayer;

function startReverseMode() {
  reverseActive    = true;
  handDetector     = new HandDetector(arOverlay.videoElement);
  frameBuffer      = new FrameBuffer(CONFIG.N_FRAMES);
  classifier       = new SignClassifier(CONFIG.CLASSIFIER_ENDPOINT, CONFIG.CONFIDENCE_THRESHOLD);
  sentenceBuilder  = new SentenceBuilder();
  ttsPlayer        = new TTSPlayer();

  glossBuffer = new GlossBuffer(CONFIG.SIGNING_PAUSE_MS, async (glosses) => {
    // Called when signing pause detected
    SnowflakeLogger.logEvent('sentence_building', { glosses });
    const sentence = await sentenceBuilder.build(glosses);
    SnowflakeLogger.logEvent('tts_start', { sentence, gloss_count: glosses.length });
    arOverlay.showSubtitle(sentence); // show what's being spoken
    await ttsPlayer.speak(sentence);
  });

  // Wire hand detector → frame buffer → classifier → gloss buffer
  handDetector.onLandmarks = async (landmarks) => {
    frameBuffer.push(landmarks);
    if (!frameBuffer.isFull()) return;

    const result = await classifier.classify(frameBuffer.getFrames());
    if (result) {
      glossBuffer.push(result.gloss);
      arOverlay.showSignLabel(result.gloss, result.confidence); // HUD overlay
    }
  };

  handDetector.start();
}

function stopReverseMode() {
  reverseActive = false;
  handDetector?.stop();
  glossBuffer?.clear();
}

// Mode toggle button handler
document.getElementById('mode-toggle').addEventListener('click', () => {
  if (reverseActive) {
    stopReverseMode();
    startForwardMode(); // resume voice-to-sign
  } else {
    stopForwardMode();  // pause microphone
    startReverseMode();
  }
});
```

---

## UI additions — mode switcher

Add to `index.html` (inside the AR container):

```html
<!-- Mode toggle button — overlaid on camera feed -->
<div class="mode-switcher">
  <button id="mode-toggle" class="mode-btn active" data-mode="forward">
    🎤 Voice → Sign
  </button>
  <button id="mode-toggle-reverse" class="mode-btn" data-mode="reverse">
    ✋ Sign → Voice
  </button>
</div>

<!-- Sign label HUD — shows current detected sign -->
<div id="sign-label" class="sign-label-hud" style="display:none">
  <span id="sign-label-text"></span>
  <span id="sign-confidence"></span>
</div>
```

Add to `style.css`:

```css
.mode-switcher {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  z-index: 20;
}

.mode-btn {
  font-size: 13px;
  padding: 8px 16px;
  border-radius: 20px;
  border: 1.5px solid rgba(255,255,255,0.3);
  background: rgba(0,0,0,0.5);
  color: #fff;
  cursor: pointer;
  backdrop-filter: blur(8px);
  transition: all 0.2s;
}

.mode-btn.active {
  background: #1D9E75;
  border-color: #1D9E75;
}

.sign-label-hud {
  position: absolute;
  top: 80px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.65);
  color: #fff;
  font-size: 28px;
  font-weight: 500;
  padding: 10px 24px;
  border-radius: 12px;
  text-align: center;
  z-index: 20;
  pointer-events: none;
}
```

---

## Running both servers

Three terminals required when running the full app:

```bash
# Terminal 1 — forward pipeline pose server (existing)
cd python
.venv/bin/uvicorn server:app --port 8000

# Terminal 2 — reverse pipeline classifier server (new)
cd python
.venv-reverse/bin/uvicorn sign_classifier:app --port 8001

# Terminal 3 — Node server
npm start
```

---

## Snowflake — New Tables for Reverse Pipeline

Run these in addition to the existing CLAUDE.md schema:

```sql
CREATE TABLE IF NOT EXISTS ACCESSLINK.PUBLIC.REVERSE_EVENTS (
  event_id       VARCHAR     DEFAULT UUID_STRING(),
  event_type     VARCHAR     NOT NULL,  -- 'sign_detected' | 'sentence_built' | 'tts_played'
  session_id     VARCHAR     NOT NULL,
  gloss          VARCHAR,               -- individual sign label
  confidence     FLOAT,                 -- classifier confidence 0.0–1.0
  gloss_sequence VARCHAR,               -- JSON array of glosses in sentence
  sentence       VARCHAR,               -- final English sentence spoken
  tts_latency_ms INTEGER,
  timestamp      TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP()
);
```

Key analytics queries:

```sql
-- Most signed words
SELECT gloss, COUNT(*) as sign_count
FROM REVERSE_EVENTS
WHERE event_type = 'sign_detected'
GROUP BY gloss
ORDER BY sign_count DESC
LIMIT 20;

-- Average confidence per gloss (find unreliable predictions)
SELECT gloss,
       ROUND(AVG(confidence) * 100, 1) AS avg_confidence_pct,
       COUNT(*) AS total_detections
FROM REVERSE_EVENTS
WHERE event_type = 'sign_detected'
GROUP BY gloss
ORDER BY avg_confidence_pct ASC;

-- Sentences spoken today
SELECT sentence, timestamp
FROM REVERSE_EVENTS
WHERE event_type = 'tts_played'
  AND timestamp > DATEADD(day, -1, CURRENT_TIMESTAMP())
ORDER BY timestamp DESC;
```

---

## Acceptance Criteria — Reverse Mode Done When

- [ ] Camera opens in reverse mode without re-requesting permission
- [ ] MediaPipe detects hands and shows landmark overlay on canvas
- [ ] Classifier endpoint (`/health`) returns `{"status": "ok"}` before starting
- [ ] Signing "HELLO" triggers a confident prediction above 0.75 threshold
- [ ] Consecutive identical predictions are deduplicated (no repeated words)
- [ ] 1.5 second pause in signing flushes the gloss buffer and triggers TTS
- [ ] Claude API converts `["hello", "name", "alex"]` → `"Hello, my name is Alex."`
- [ ] ElevenLabs speaks the sentence within 1 second of sentence completion
- [ ] Web Speech API fallback fires when ElevenLabs key is missing
- [ ] Sign label HUD shows current detected sign + confidence on screen
- [ ] Snowflake receives `sign_detected` and `tts_played` events per session

---

## Performance Targets — Reverse Mode

| Metric                             | Target      |
|------------------------------------|-------------|
| Hand detection latency             | < 50ms      |
| Classifier round-trip (local)      | < 200ms     |
| Gloss → sentence (Claude API)      | < 800ms     |
| Sentence → audio starts (ElevenLabs) | < 300ms   |
| Full end-to-end (sign → speech)    | < 2s        |
| Minimum confidence threshold       | 0.75        |
| WLASL vocabulary covered           | 100 words   |

---

## Known Limitations (acknowledge in pitch)

- WLASL100 covers 100 words — not the full ASL vocabulary
- The TGCN model was trained on multiple signers but may have lower accuracy for unusual hand sizes, skin tones, or signing styles not in WLASL
- Dynamic signs (multi-movement) require exactly 30 frames — static or partial signs may produce low-confidence results correctly rejected by the threshold
- ASL grammar is not reconstructed — the Claude API receives English-ordered glosses and produces grammatical English, not authentic ASL-to-English translation
- The classifier server requires Python running locally — not a pure browser solution

---

## Demo Script — Reverse Mode

Practice these signs in sequence. Each word is in WLASL100:

1. **HELLO → NAME → ALEX** → speaks: *"Hello, my name is Alex."*
2. **HELP → PLEASE** → speaks: *"Please help me."*
3. **WATER → NEED** → speaks: *"I need water."*
4. **SCHOOL → GO** → speaks: *"I'm going to school."*
5. **THANK → YOU** → speaks: *"Thank you."*

---

## Key Resources — Reverse Pipeline

| Resource | URL |
|---|---|
| WLASL TGCN model (Hugging Face) | https://huggingface.co/sharonn18/tgcn-wlasl |
| WLASL dataset + original TGCN code | https://github.com/dxli94/WLASL |
| MediaPipe Hands (Python) | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |
| MediaPipe Hands (JS / npm) | https://www.npmjs.com/package/@mediapipe/hands |
| ElevenLabs Flash v2.5 | https://elevenlabs.io/docs/api-reference/text-to-speech |
| Anthropic Claude API | https://docs.anthropic.com/en/api/getting-started |
| Google ASL Signs dataset (Kaggle) | https://www.kaggle.com/competitions/asl-signs |
| Sign Language MNIST (fingerspelling) | https://www.kaggle.com/datasets/datamunge/sign-language-mnist |
