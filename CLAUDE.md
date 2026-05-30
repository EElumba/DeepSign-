# AccessLink — AI Coding Assistant Context
> Paste this file into your AI coding assistant (Cursor, Windsurf, Claude Code, etc.) as persistent project context.
> Last updated: May 2026 · Hackathon MVP scope

---

## What is AccessLink?

AccessLink is a browser-based accessibility tool that translates live spoken audio into real-time ASL sign language animations — displayed as an AR overlay on a live camera feed. It simulates the experience of looking through smart AR glasses (like Ray-Ban Meta) that caption and sign the world around a Deaf user.

**The core value proposition:** A Deaf person opens AccessLink in a browser tab. Someone nearby speaks. Within 500ms, a signing avatar appears on screen performing each word in ASL — no interpreter, no app install, no hardware required.

---

## MVP Scope

**In scope:**
- Live microphone → Deepgram STT → word tokens → sign animation → AR canvas overlay
- Browser-only, no backend server, no login, no install
- 50-word ASL sign vocabulary for demo day
- Snowflake session logging (async, non-blocking)
- Chrome desktop as the primary target

**Out of scope (post-MVP roadmap):**
- Sign language → voice (reverse pipeline for mute users)
- Blind user scene description / narration
- ASL grammar reordering (MVP uses Signed Exact English — word-for-word order)
- HamNoSys / SignWriting notation engine
- Physical AR glasses hardware integration
- Mobile app
- Multi-language sign support (BSL, DGS, etc.)

---

## Pipeline — Full Overview

```
[Microphone]
     │  getUserMedia({audio:true}) → MediaRecorder 250ms blobs
     ▼
[Stage 1: Audio Capture]
     │  WebSocket stream → Deepgram Nova-3
     ▼
[Stage 2: Transcription]  ──── interim results ──→ [Caption bar update]
     │  is_final:true results only
     ▼
[Stage 3: Normalisation]
     │  lowercase · strip punctuation · split on whitespace
     ▼
[Stage 4: Dictionary Lookup]
     │  word → signId  (or null → skip + log to Snowflake)
     ▼
[Stage 5: Sign Queue (FIFO)]
     │  array buffer · one sign dequeued at a time
     ▼
[Stage 6: Animation Player]
     │  load asset · play · await onSignComplete · 400ms gap · dequeue next
     ▼
[Stage 7: AR Overlay Display]
     │  requestAnimationFrame loop
     │  camera feed (<video>) + avatar frame + caption bar → composited canvas
     ▼
[User sees: live camera + signing avatar + live captions]

                         └──── async ────→ [Snowflake: session events log]
```

---

## Architecture — Key Design Decisions

### No translation to HamNoSys or SignWriting
The MVP does NOT use any linguistic notation system. Words map directly to pre-built animation assets. This is called "Signed Exact English" (SEE) — it signs English words in order, not grammatical ASL. This is intentional for the hackathon scope. Post-MVP, an ASL grammar layer would reorder and drop words to match true ASL sentence structure before the sign queue.

### Interim vs final transcripts
Deepgram returns two types of results:
- `is_final: false` — partial, unconfirmed guesses. Used ONLY for the live caption bar preview. Never trigger sign animations.
- `is_final: true` — confirmed transcript of a complete utterance. These enter the sign pipeline.

### FIFO queue as the coordination layer
The queue decouples fast speech input from slow animation playback. Speech can produce 5 words in 2 seconds; each sign animation takes 1–2 seconds to play. The queue buffers the backlog and ensures signs always play in spoken order.

### Web Speech API fallback
If `CONFIG.DEEPGRAM_API_KEY` is empty or blank, the system automatically falls back to the browser's native `SpeechRecognition` API. This means the demo always works, even without an API key — essential for hackathon reliability.

### Signed Exact English (SEE) vs ASL
ASL is not English with hands. It has its own grammar: topic-comment structure, no articles, no linking verbs, spatial grammar. The MVP signs words in English order (SEE). If asked by judges or community members, acknowledge this and frame proper ASL grammar as a post-MVP enhancement.

---

## File Structure

```
accesslink/
├── index.html              # Entry point — loads all modules
├── style.css               # AR layout: stacked video + canvas, fullscreen
├── config.js               # API keys, timing constants, feature flags
├── main.js                 # Wires all modules together, starts session
├── modules/
│   ├── AudioCapture.js     # Mic → MediaRecorder → Deepgram WebSocket
│   ├── WordQueue.js        # FIFO sign queue with inter-sign delay
│   ├── SignPlayer.js       # signId → animation trigger → onSignComplete
│   ├── SignDictionary.js   # word string → signId map (50+ entries)
│   └── AROverlay.js        # Camera feed + canvas compositor
├── animations/
│   └── [sign assets]       # GLB / Rive / Lottie / MP4 files per sign
└── logging/
    └── SnowflakeLogger.js  # Async session event batching → Snowflake REST
```

---

## Module Specifications

### `AudioCapture.js`

Responsibilities:
- Open microphone with `getUserMedia({ audio: true })`
- Create `MediaRecorder` emitting 250ms `audio/webm` blobs
- Connect to Deepgram WebSocket, stream blobs
- Parse incoming messages, emit two events:
  - `onInterim(text)` — for caption bar preview
  - `onFinal(text)` — for sign pipeline
- Handle WebSocket disconnection with exponential backoff reconnect

```js
// Core Deepgram connection
const socket = new WebSocket('wss://api.deepgram.com/v1/listen', [
  'token',
  CONFIG.DEEPGRAM_API_KEY,
]);

socket.onopen = () => {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    recorder.ondataavailable = e => {
      if (socket.readyState === WebSocket.OPEN) socket.send(e.data);
    };
    recorder.start(250);
  });
};

socket.onmessage = msg => {
  const data = JSON.parse(msg.data);
  const transcript = data?.channel?.alternatives?.[0]?.transcript;
  const isFinal = data?.is_final;
  if (!transcript) return;
  if (isFinal) onFinal(transcript);
  else onInterim(transcript);
};
```

### `WordQueue.js`

Responsibilities:
- Accept a transcript string from `AudioCapture.onFinal`
- Normalise: `text.toLowerCase().replace(/[^\w\s]/g, '').trim()`
- Tokenise: `text.split(/\s+/).filter(Boolean)`
- For each token, call `SignDictionary.lookup(word)`
  - If match found → push `signId` to internal FIFO array
  - If no match → call `SnowflakeLogger.logGap(word)`
- When queue transitions from empty → non-empty, call `onWordReady()`
- After each sign completes, wait `CONFIG.SIGN_DELAY_MS` then dequeue next

```js
// Queue drain loop
async function drainQueue() {
  while (queue.length > 0) {
    const signId = queue.shift();
    await SignPlayer.play(signId);
    await sleep(CONFIG.SIGN_DELAY_MS);
  }
  isPlaying = false;
}
```

### `SignPlayer.js`

Responsibilities:
- Accept a `signId` string
- Load/retrieve the corresponding animation asset
- Trigger playback on the canvas
- Return a Promise that resolves when the animation completes
- Expose `onSignStart(signId)` and `onSignComplete(signId)` hooks (used by caption bar highlighting)

Required public interface — must be implemented regardless of animation library chosen:
```js
class SignPlayer {
  constructor(canvasElement) { /* init animation lib on canvas */ }

  async play(signId) {
    this.emit('signStart', signId);
    await this._triggerAnimation(signId);
    this.emit('signComplete', signId);
  }

  // Implementation varies by animation library:
  // Video:    videoEl.src = `animations/${signId}.mp4`; await videoEl.play(); await onEnded
  // Three.js: mixer.clipAction(clips[signId]).play(); await onLoop
  // Rive:     rive.play(signId); await onStateChange('complete')
  // Lottie:   anim.goToAndPlay(segments[signId]); await onComplete
  _triggerAnimation(signId) { /* ... */ }
}
```

### `SignDictionary.js`

A plain JS object exported as the source of truth for the sign vocabulary. Keys are lowercase normalised words. Values are signId strings that map to asset filenames.

```js
export const SignDictionary = {
  // Greetings
  'hello':        'sign_hello',
  'goodbye':      'sign_goodbye',
  'hi':           'sign_hello',

  // Courtesy
  'thank':        'sign_thank_you',
  'thanks':       'sign_thank_you',
  'please':       'sign_please',
  'sorry':        'sign_sorry',
  'welcome':      'sign_welcome',

  // Responses
  'yes':          'sign_yes',
  'no':           'sign_no',
  'understand':   'sign_understand',
  'help':         'sign_help',

  // Pronouns
  'i':            'sign_i',
  'you':          'sign_you',
  'my':           'sign_my',
  'your':         'sign_your',
  'we':           'sign_we',

  // Common verbs
  'need':         'sign_need',
  'want':         'sign_want',
  'eat':          'sign_eat',
  'drink':        'sign_drink',
  'go':           'sign_go',
  'come':         'sign_come',
  'know':         'sign_know',
  'like':         'sign_like',
  'love':         'sign_love',

  // Question words
  'what':         'sign_what',
  'where':        'sign_where',
  'when':         'sign_when',
  'who':          'sign_who',
  'how':          'sign_how',
  'why':          'sign_why',

  // Common nouns
  'name':         'sign_name',
  'water':        'sign_water',
  'food':         'sign_food',
  'home':         'sign_home',
  'school':       'sign_school',
  'work':         'sign_work',
  'doctor':       'sign_doctor',
  'friend':       'sign_friend',
  'family':       'sign_family',
  'time':         'sign_time',
  'today':        'sign_today',

  // Descriptors
  'good':         'sign_good',
  'bad':          'sign_bad',
  'happy':        'sign_happy',
  'sad':          'sign_sad',
  'hot':          'sign_hot',
  'cold':         'sign_cold',
  'big':          'sign_big',
  'small':        'sign_small',

  // Demo script words
  'nice':         'sign_nice',
  'meet':         'sign_meet',
  'alex':         'sign_name',
};

export function lookup(word) {
  return SignDictionary[word.toLowerCase()] || null;
}
```

### `AROverlay.js`

Responsibilities:
- Create and stack `<video>` (camera) and `<canvas>` (overlay) elements via CSS
- Start camera stream via `getUserMedia({ video: { facingMode: CONFIG.CAMERA_FACING } })`
- Run `requestAnimationFrame` loop calling `render()`
- `render()` draws:
  1. Camera frame → canvas (background)
  2. Current animation frame from SignPlayer → canvas (foreground, bottom-right)
  3. Caption bar text → canvas (bottom-centre, semi-transparent background)
  4. Currently-signing word → highlighted in `CONFIG.CAPTION_HIGHLIGHT_COLOR`

### `SnowflakeLogger.js`

Responsibilities:
- Buffer events in memory (max 20 events or 10 seconds, whichever comes first)
- Flush batch to Snowflake REST API asynchronously
- Never block the sign pipeline — all calls are fire-and-forget with silent catch

Events logged:
```js
// Session start
{ event: 'session_start', session_id, timestamp, user_agent }

// Sign played
{ event: 'sign_played', session_id, word, sign_id, stt_latency_ms, anim_latency_ms, timestamp }

// Gap (word not in dictionary)
{ event: 'sign_gap', session_id, word, timestamp }

// Session end
{ event: 'session_end', session_id, duration_ms, total_words, matched_words, gap_words, timestamp }
```

Snowflake REST API call:
```js
async function flush(events) {
  try {
    await fetch(`https://${CONFIG.SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/statements`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CONFIG.SNOWFLAKE_JWT}`,
        'Content-Type': 'application/json',
        'X-Snowflake-Authorization-Token-Type': 'KEYPAIR_JWT',
      },
      body: JSON.stringify({
        statement: buildInsertSQL(events),
        database: CONFIG.SNOWFLAKE_DATABASE,
        schema: CONFIG.SNOWFLAKE_SCHEMA,
        warehouse: CONFIG.SNOWFLAKE_WAREHOUSE,
      }),
    });
  } catch (e) {
    console.warn('Snowflake log failed (non-blocking):', e.message);
  }
}
```

---

## Configuration (`config.js`)

```js
export const CONFIG = {
  // APIs
  DEEPGRAM_API_KEY:       '',       // Leave empty → Web Speech API fallback
  SNOWFLAKE_ACCOUNT:      '',       // e.g. 'xy12345.us-east-1'
  SNOWFLAKE_JWT:          '',       // JWT for key-pair auth
  SNOWFLAKE_DATABASE:     'ACCESSLINK',
  SNOWFLAKE_SCHEMA:       'PUBLIC',
  SNOWFLAKE_WAREHOUSE:    'COMPUTE_WH',

  // Pipeline timing
  SIGN_DELAY_MS:          400,      // Gap between signs (ms)
  AUDIO_CHUNK_MS:         250,      // MediaRecorder chunk size

  // Display
  SHOW_CAPTIONS:          true,
  CAMERA_FACING:          'user',   // 'user' (front) or 'environment' (rear)
  AVATAR_POSITION:        'bottom-right',
  AVATAR_SCALE:           0.4,
  CAPTION_HIGHLIGHT_COLOR:'#1D9E75',

  // Logging
  LOG_BATCH_SIZE:         20,
  LOG_FLUSH_INTERVAL_MS:  10000,

  // Debug
  DEBUG_MODE:             false,    // Logs queue state + sign lookups to console
};
```

---

## CSS Layout (Critical — do not change)

The AR effect depends on precise stacking of video and canvas:

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #000;
  overflow: hidden;
  width: 100vw;
  height: 100vh;
}

.ar-container {
  position: relative;
  width: 100vw;
  height: 100vh;
}

.ar-container video {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  object-fit: cover;
  transform: scaleX(-1); /* mirror front camera */
}

.ar-container canvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}

.start-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.7);
  z-index: 10;
}

.start-btn {
  font-size: 18px;
  padding: 16px 40px;
  border-radius: 8px;
  border: none;
  background: #1D9E75;
  color: #fff;
  cursor: pointer;
}
```

---

## Web Speech API Fallback

Used automatically when `CONFIG.DEEPGRAM_API_KEY` is empty:

```js
function startWebSpeechFallback(onFinal, onInterim) {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.error('No speech recognition available');
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onresult = e => {
    for (const result of e.results) {
      const transcript = result[0].transcript;
      if (result.isFinal) onFinal(transcript);
      else onInterim(transcript);
    }
  };
  recognition.onerror = e => console.warn('Web Speech error:', e.error);
  recognition.onend = () => recognition.start(); // auto-restart
  recognition.start();
}
```

---

## Animation Library Integration

Whichever library is chosen, wrap it to match this interface exactly:

```js
// The contract SignPlayer must fulfil — library-agnostic
class SignPlayerInterface {
  async play(signId)             // triggers animation, returns Promise
  on('signStart', cb)            // called when animation begins
  on('signComplete', cb)         // called when animation finishes
  isPlaying()                    // returns boolean
  stop()                         // cancels current animation immediately
}
```

### Option 1 — Video clips (fastest to implement)
```js
async _triggerAnimation(signId) {
  return new Promise(resolve => {
    this.video.src = `animations/${signId}.mp4`;
    this.video.onended = resolve;
    this.video.play();
  });
}
```

### Option 2 — Three.js + Ready Player Me GLB
```js
async _triggerAnimation(signId) {
  const clip = this.clips[signId];
  const action = this.mixer.clipAction(clip);
  action.reset().setLoop(THREE.LoopOnce).play();
  return new Promise(resolve => {
    this.mixer.addEventListener('finished', resolve, { once: true });
  });
}
```

### Option 3 — Rive state machine
```js
async _triggerAnimation(signId) {
  this.rive.play(signId);
  return new Promise(resolve => {
    this.rive.on(EventType.StateChange, e => {
      if (e.data.includes('complete')) resolve();
    });
  });
}
```

### Option 4 — Lottie
```js
async _triggerAnimation(signId) {
  const [start, end] = this.segments[signId];
  this.anim.playSegments([start, end], true);
  return new Promise(resolve => {
    this.anim.addEventListener('complete', resolve, { once: true });
  });
}
```

---

## Snowflake Table Schema

```sql
CREATE DATABASE IF NOT EXISTS ACCESSLINK;
CREATE SCHEMA IF NOT EXISTS ACCESSLINK.PUBLIC;

CREATE TABLE IF NOT EXISTS ACCESSLINK.PUBLIC.SIGN_EVENTS (
  event_id       VARCHAR     DEFAULT UUID_STRING(),
  event_type     VARCHAR     NOT NULL,
  session_id     VARCHAR     NOT NULL,
  word           VARCHAR,
  sign_id        VARCHAR,
  stt_latency_ms INTEGER,
  anim_latency_ms INTEGER,
  matched        BOOLEAN,
  timestamp      TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS ACCESSLINK.PUBLIC.SESSIONS (
  session_id     VARCHAR     PRIMARY KEY,
  started_at     TIMESTAMP_TZ,
  ended_at       TIMESTAMP_TZ,
  duration_ms    INTEGER,
  total_words    INTEGER,
  matched_words  INTEGER,
  gap_words      INTEGER,
  user_agent     VARCHAR
);
```

Useful analytics queries:

```sql
-- Top 20 words with no sign (expand dictionary here first)
SELECT word, COUNT(*) AS missed_count
FROM SIGN_EVENTS
WHERE event_type = 'sign_gap'
GROUP BY word
ORDER BY missed_count DESC
LIMIT 20;

-- Average end-to-end latency per session
SELECT session_id,
       AVG(stt_latency_ms)  AS avg_stt_ms,
       AVG(anim_latency_ms) AS avg_anim_ms,
       AVG(stt_latency_ms + anim_latency_ms) AS avg_total_ms
FROM SIGN_EVENTS
WHERE event_type = 'sign_played'
GROUP BY session_id;

-- Match rate over time
SELECT DATE_TRUNC('hour', timestamp) AS hour,
       COUNT_IF(matched) / COUNT(*) * 100 AS match_pct
FROM SIGN_EVENTS
WHERE event_type IN ('sign_played','sign_gap')
GROUP BY hour
ORDER BY hour;
```

---

## Demo Script (practice before presenting)

Say these phrases to demonstrate the full pipeline:

1. `"Hello my name is Alex"` — tests greeting + name
2. `"Nice to meet you"` — tests courtesy words
3. `"Can you help me please"` — tests request + courtesy
4. `"Yes I understand"` — tests affirmative response
5. `"I need water"` — tests need + noun
6. `"Thank you goodbye"` — tests closing

Every word in these phrases is in the sign dictionary. Practice until all signs play smoothly with no gaps.

---

## Acceptance Criteria — MVP Done When

- [ ] Microphone opens on first click, no repeated permission prompts
- [ ] First caption word appears within 300ms of speech
- [ ] Each recognised word triggers its sign animation or silently skips if unknown
- [ ] Signs play one at a time in spoken order — never simultaneously
- [ ] Camera feed is visible underneath the avatar at all times
- [ ] Caption bar highlights the word currently being signed
- [ ] Snowflake receives at least one `sign_played` row per session
- [ ] Demo script ("Hello my name is Alex…") completes with zero failed signs
- [ ] Web Speech API fallback works when no Deepgram key is configured
- [ ] Runs fully in Chrome desktop — no server, no install, no login

---

## Performance Targets

| Metric                        | Target     |
|-------------------------------|------------|
| Speech → first caption word   | < 300ms    |
| Caption → first sign starts   | < 100ms    |
| Full end-to-end latency       | < 500ms    |
| Page load to live demo        | < 2s       |
| Signs in launch dictionary    | 50+        |
| Browser support               | Chrome 110+ |

---

## Post-MVP Roadmap (mention in pitch, don't build now)

1. **ASL grammar layer** — NLP reordering to produce grammatical ASL, not SEE
2. **HamNoSys engine** — procedural sign generation for unlimited vocabulary
3. **Reverse pipeline** — camera → MediaPipe Hands → sign classifier → voice (for mute users)
4. **Blind user mode** — scene description via AI vision narrated through bone-conduction audio
5. **Fingerspelling fallback** — spell unknown words letter by letter
6. **Hardware port** — Ray-Ban Meta / Snap Spectacles integration
7. **Mobile app** — React Native wrapper
8. **Expanded vocabulary** — 3,000+ signs covering full ASL dictionary

---

## Key Resources

| Resource | URL |
|---|---|
| Deepgram docs (streaming) | https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio |
| Deepgram free $200 credit | https://console.deepgram.com |
| Ready Player Me avatars | https://readyplayer.me |
| Mixamo free animations | https://mixamo.com |
| Three.js GLTF loader | https://threejs.org/docs/#examples/en/loaders/GLTFLoader |
| Rive runtime | https://rive.app/community |
| LottieFiles | https://lottiefiles.com |
| ASL sign reference | https://www.handspeak.com |
| ASL-LEX database | https://asl-lex.org |
| Snowflake REST API | https://docs.snowflake.com/en/developer-guide/sql-api/guide |
| MediaPipe Hands (post-MVP) | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |
