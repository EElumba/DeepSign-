import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { randomUUID } from 'crypto';
import fetch from 'node-fetch';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';
import OpenAI from 'openai';

const __dirname = dirname(fileURLToPath(import.meta.url));

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
if (!DEEPGRAM_API_KEY) {
  console.warn('[Speech] DEEPGRAM_API_KEY not set — Speak→Sign audio capture will be unavailable.');
}

// Strip trailing slashes so `${POSE_SERVER_URL}/pose` never becomes `//pose`
// (a double slash 404s on FastAPI/Starlette).
const POSE_SERVER_URL = (process.env.POSE_SERVER_URL || 'http://localhost:8000').replace(/\/+$/, '');
const FINAL_FLUSH_DELAY_MS = Number(process.env.FINAL_FLUSH_DELAY_MS || 120);
const POSE_TIMEOUT_MS = Number(process.env.POSE_TIMEOUT_MS || 30000);
const ROOM_IDLE_TTL_MS = Number(process.env.ROOM_IDLE_TTL_MS || 60000);
const ROOM_ID_PATTERN = /^[a-zA-Z0-9_-]{6,80}$/;
const DEMO_POSE_ENABLED = process.env.DEMO_POSE_ENABLED !== '0';
const ALLOW_CUSTOM_DEMO_POSE = process.env.ALLOW_CUSTOM_DEMO_POSE === '1';

const DEMO_PHRASES = Object.freeze([
  { id: 'hello', label: 'Hello', text: 'hello nice meet you' },
  { id: 'help', label: 'Need help', text: 'please help me' },
  { id: 'family', label: 'Family', text: 'my family learn sign' },
  { id: 'yes-no', label: 'Yes / no', text: 'yes no' },
]);

const deepgram = DEEPGRAM_API_KEY ? createClient(DEEPGRAM_API_KEY) : null;

// --- English -> ASL gloss via OpenAI -------------------------------------- //
// Small/fast model keeps latency low. If no key is set, we sign the raw
// transcript (the Python glosser still lemmatizes + fingerspells unknowns), so
// the app degrades gracefully.
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const GLOSS_TIMEOUT_MS = Number(process.env.GLOSS_TIMEOUT_MS || 1500);
const GLOSS_CACHE_SIZE = Math.max(0, Number(process.env.GLOSS_CACHE_SIZE || 128));

const openai = OPENAI_API_KEY
  ? new OpenAI({ apiKey: OPENAI_API_KEY, timeout: GLOSS_TIMEOUT_MS, maxRetries: 0 })
  : null;

if (!openai) {
  console.warn('[Gloss] OPENAI_API_KEY not set — signing raw transcript (no ASL gloss).');
} else {
  console.log(`[Gloss] ASL glossing enabled with model "${OPENAI_MODEL}".`);
}

// --- ElevenLabs TTS (Sign→Speak mode) ------------------------------------- //
const ELEVENLABS_API_KEY  = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM';

if (!ELEVENLABS_API_KEY) {
  console.warn('[TTS] ELEVENLABS_API_KEY not set — /api/tts will return 503.');
} else {
  console.log('[TTS] ElevenLabs TTS enabled.');
}

const GLOSS_TO_ENGLISH_PROMPT =
  'You are an ASL interpreter. Convert ASL gloss notation into natural spoken English. ' +
  'ASL gloss uses base word forms, no articles, and topic-comment word order. ' +
  'Output only the English sentence — no commentary, no extra punctuation.';

const GLOSS_SYSTEM_PROMPT =
  'You are an expert American Sign Language (ASL) interpreter. Convert the ' +
  "English text into ASL gloss. Rules: use ASL grammar and word order " +
  '(topic-comment, time first); drop articles (a/an/the) and forms of "to be" ' +
  '(is/am/are/was/were); use uninflected base words (no -ing/-ed/plural -s); ' +
  'keep WH question words (what/who/where/when/why/how) in natural ASL position; ' +
  'expand contractions. Output ONLY the gloss as space-separated UPPERCASE words ' +
  'with no punctuation and no commentary.';

// Cache so repeated phrases ("hello", "thank you") cost nothing after the first.
const glossCache = new Map();

const DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

function expandDigitsForSigning(text) {
  return text.replace(/\d+/g, (digits) => (
    digits
      .split('')
      .map((digit) => DIGIT_WORDS[Number(digit)])
      .join(' ')
  ));
}

function normalizePhraseText(value) {
  return String(value || '').trim().replace(/\s+/g, ' ').toLowerCase();
}

function findDemoPhrase({ phraseId, text }) {
  if (phraseId) {
    const byId = DEMO_PHRASES.find((phrase) => phrase.id === String(phraseId));
    if (byId) return byId;
  }

  const normalized = normalizePhraseText(text);
  if (!normalized) return null;
  return DEMO_PHRASES.find((phrase) => normalizePhraseText(phrase.text) === normalized) || null;
}

async function englishToAslGloss(text) {
  if (!openai) return text;
  const key = text.toLowerCase().trim();
  if (glossCache.has(key)) {
    const cached = glossCache.get(key);
    glossCache.delete(key);
    glossCache.set(key, cached);
    return cached;
  }
  try {
    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0,
      max_tokens: 60,
      messages: [
        { role: 'system', content: GLOSS_SYSTEM_PROMPT },
        { role: 'user', content: text },
      ],
    });
    const gloss = completion.choices?.[0]?.message?.content?.trim();
    const result = gloss && gloss.length ? gloss : text;
    if (GLOSS_CACHE_SIZE > 0) {
      glossCache.set(key, result);
      while (glossCache.size > GLOSS_CACHE_SIZE) {
        glossCache.delete(glossCache.keys().next().value);
      }
    }
    return result;
  } catch (err) {
    console.error('[Gloss] OpenAI failed, using raw text:', err.message);
    return text; // graceful fallback — never block the avatar on the API
  }
}

const app = express();
app.use(express.json({ limit: '5mb' }));

// Room-local state. A room represents one private conversation. All transcripts,
// pose blobs, and control messages stay inside its display client set.
const rooms = new Map();

function createRoomId() {
  return randomUUID().replace(/-/g, '');
}

function normalizeRoomId(value) {
  const roomId = String(value || '').trim();
  return ROOM_ID_PATTERN.test(roomId) ? roomId : '';
}

function roomPath(pathname, roomId) {
  return `${pathname}?room=${encodeURIComponent(roomId)}`;
}

function roomFromHttpRequest(req) {
  return normalizeRoomId(req.query?.room);
}

function sendRoomPage(req, res, pathname, filename) {
  const roomId = roomFromHttpRequest(req);
  if (!roomId) {
    res.redirect(302, roomPath(pathname, createRoomId()));
    return;
  }
  res.sendFile(join(__dirname, filename));
}

// Landing/migration route: every fresh visit gets a private room by default.
app.get('/', (req, res) => {
  res.redirect(302, roomPath('/speak', createRoomId()));
});

// Speaker/controller page (phone or laptop): captures the mic and streams audio.
app.get('/speak', (req, res) => {
  sendRoomPage(req, res, '/speak', 'index.html');
});

// Programmatic room creation hook for future QR/pairing flows.
app.get('/api/sessions/new', (req, res) => {
  const roomId = createRoomId();
  res.json({
    roomId,
    speakUrl: roomPath('/speak', roomId),
    glassesUrl: roomPath('/glasses', roomId),
    audioWsPath: `/ws/audio/${roomId}`,
    displayWsPath: `/ws/display/${roomId}`,
  });
});

async function probePoseServer() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1200);
  try {
    const response = await fetch(`${POSE_SERVER_URL}/health`, { signal: controller.signal });
    return {
      configured: true,
      reachable: response.ok,
      status: response.ok ? 'ok' : `http_${response.status}`,
    };
  } catch (err) {
    return {
      configured: true,
      reachable: false,
      status: err.name === 'AbortError' ? 'timeout' : 'unreachable',
    };
  } finally {
    clearTimeout(timeout);
  }
}

app.get('/api/health', async (req, res) => {
  res.json({
    status: 'ok',
    services: {
      deepgram: {
        configured: Boolean(DEEPGRAM_API_KEY),
        status: DEEPGRAM_API_KEY ? 'configured' : 'missing_key',
      },
      openai: {
        configured: Boolean(OPENAI_API_KEY),
        status: OPENAI_API_KEY ? 'configured' : 'fallback_raw_text',
      },
      tts: {
        configured: Boolean(ELEVENLABS_API_KEY),
        status: ELEVENLABS_API_KEY ? 'configured' : 'visual_only',
      },
      pose: await probePoseServer(),
      demo: {
        configured: DEMO_POSE_ENABLED,
        status: DEMO_POSE_ENABLED ? 'available' : 'disabled',
        phraseCount: DEMO_PHRASES.length,
      },
    },
    rooms: {
      active: rooms.size,
    },
  });
});

app.get('/api/demo/phrases', (req, res) => {
  res.json({
    enabled: DEMO_POSE_ENABLED,
    allowCustom: ALLOW_CUSTOM_DEMO_POSE,
    phrases: DEMO_POSE_ENABLED ? DEMO_PHRASES : [],
  });
});

app.post('/api/demo/pose', async (req, res) => {
  if (!DEMO_POSE_ENABLED) {
    return res.status(404).json({ error: 'Demo pose generation is disabled.' });
  }

  const roomId = normalizeRoomId(req.body?.roomId || req.query?.room);
  if (!roomId) return res.status(400).json({ error: 'A valid roomId is required.' });

  let phrase = findDemoPhrase({
    phraseId: req.body?.phraseId,
    text: req.body?.text,
  });

  if (!phrase && ALLOW_CUSTOM_DEMO_POSE) {
    const text = String(req.body?.text || '').trim().replace(/\s+/g, ' ');
    if (text.length > 0 && text.length <= 120) {
      phrase = { id: 'custom', label: 'Custom', text };
    }
  }

  if (!phrase) {
    return res.status(400).json({ error: 'Choose one of the available demo phrases.' });
  }

  const room = getRoom(roomId);
  const generation = room.poseGeneration;
  const signableText = expandDigitsForSigning(phrase.text);

  try {
    const gloss = expandDigitsForSigning(await englishToAslGloss(signableText));
    broadcastToRoom(room, JSON.stringify({
      type: 'transcript',
      roomId: room.id,
      source: 'demo',
      text: phrase.text,
      is_final: true,
    }));

    const arrayBuffer = await generatePose(gloss);
    if (!arrayBuffer) {
      const message = 'Demo signing is unavailable. Make sure the Python pose server is running.';
      broadcastError(room, message, 'pose');
      return res.status(503).json({ error: message });
    }

    if (generation === room.poseGeneration) {
      broadcastToRoom(room, Buffer.from(arrayBuffer));
    }

    return res.json({
      ok: true,
      roomId: room.id,
      phraseId: phrase.id,
      text: phrase.text,
      gloss,
      displayClients: room.displayClients.size,
      discarded: generation !== room.poseGeneration,
    });
  } catch (err) {
    console.error('[Demo] pose generation failed:', err.message);
    const message = 'Demo signing failed.';
    broadcastError(room, message, 'pose');
    return res.status(500).json({ error: message });
  } finally {
    releaseRoom(room);
  }
});

// Glasses display page: the Meta Ray-Ban Display Web App URL. Use the same
// room query param as /speak to pair the two clients.
app.get('/glasses', (req, res) => {
  sendRoomPage(req, res, '/glasses', 'glasses.html');
});

// Backwards-compatible alias for older bookmarks.
app.get('/room/:roomId', (req, res) => {
  const roomId = normalizeRoomId(req.params.roomId);
  if (!roomId) return res.status(400).send('Invalid room id');
  res.redirect(302, roomPath('/speak', roomId));
});

// Web App manifest + favicon (the glasses runtime requires a PNG icon; SVG is
// not supported).
app.get('/manifest.webmanifest', (req, res) => {
  res.type('application/manifest+json').sendFile(join(__dirname, 'manifest.webmanifest'));
});
app.get('/icon.png', (req, res) => {
  res.sendFile(join(__dirname, 'public', 'icon.png'), (err) => {
    if (err) res.status(404).end();
  });
});

// Sign->Speak REST endpoints.

// Proxy landmark frames to the Python recognizer
app.post('/api/recognize', async (req, res) => {
  try {
    const r = await fetch(`${POSE_SERVER_URL}/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (!r.ok) return res.status(r.status).json({ error: 'Recognition failed' });
    res.json(await r.json());
  } catch (err) {
    console.error('[Recognize]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Convert ASL gloss to natural English via OpenAI
app.post('/api/gloss-to-english', async (req, res) => {
  const { gloss } = req.body || {};
  if (!gloss) return res.status(400).json({ error: 'gloss required' });
  if (!openai) return res.json({ text: gloss }); // graceful fallback

  try {
    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.3,
      max_tokens: 80,
      messages: [
        { role: 'system', content: GLOSS_TO_ENGLISH_PROMPT },
        { role: 'user', content: gloss },
      ],
    });
    res.json({ text: completion.choices[0].message.content.trim() });
  } catch (err) {
    console.error('[Gloss→EN]', err.message);
    res.json({ text: gloss }); // return raw gloss on failure
  }
});

// ElevenLabs TTS proxy — keeps the API key server-side
app.post('/api/tts', async (req, res) => {
  const { text } = req.body || {};
  if (!text) return res.status(400).json({ error: 'text required' });
  if (!ELEVENLABS_API_KEY) return res.status(503).json({ error: 'ELEVENLABS_API_KEY not configured' });

  try {
    const r = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        }),
      }
    );
    if (!r.ok) {
      const errText = await r.text();
      console.error('[TTS] ElevenLabs error:', errText);
      return res.status(500).json({ error: 'TTS request failed' });
    }
    res.set('Content-Type', 'audio/mpeg');
    r.body.pipe(res);
  } catch (err) {
    console.error('[TTS]', err.message);
    res.status(500).json({ error: err.message });
  }
});

const server = createServer(app);
const wss = new WebSocketServer({ server });

function getRoom(roomId) {
  let room = rooms.get(roomId);
  if (!room) {
    room = {
      id: roomId,
      displayClients: new Set(),
      audioSessions: new Set(),
      poseGeneration: 0,
      cleanupTimer: null,
      createdAt: Date.now(),
      lastActivityAt: Date.now(),
    };
    rooms.set(roomId, room);
    console.log(`[Room ${roomId}] created (${rooms.size} active room(s)).`);
  }
  room.lastActivityAt = Date.now();
  if (room.cleanupTimer) {
    clearTimeout(room.cleanupTimer);
    room.cleanupTimer = null;
  }
  return room;
}

function roomHasClients(room) {
  return room.displayClients.size > 0 || room.audioSessions.size > 0;
}

function releaseRoom(room) {
  room.lastActivityAt = Date.now();
  if (roomHasClients(room) || room.cleanupTimer) return;

  room.cleanupTimer = setTimeout(() => {
    const current = rooms.get(room.id);
    if (!current || roomHasClients(current)) return;
    rooms.delete(room.id);
    console.log(`[Room ${room.id}] cleaned up (${rooms.size} active room(s)).`);
  }, ROOM_IDLE_TTL_MS);
}

function parseWebSocketRequest(request) {
  const url = new URL(request.url || '/', 'http://localhost');
  const parts = url.pathname.split('/').filter(Boolean);

  let role = 'display';
  let roomId = '';

  if (parts[0] === 'ws' && (parts[1] === 'audio' || parts[1] === 'display')) {
    role = parts[1];
    roomId = normalizeRoomId(parts[2]) || normalizeRoomId(url.searchParams.get('room'));
  } else if (parts[0] === 'audio') {
    // Compatibility: older clients can move from /audio to /audio?room=<id>.
    role = 'audio';
    roomId = normalizeRoomId(parts[1]) || normalizeRoomId(url.searchParams.get('room'));
  } else {
    // Compatibility: display clients may still connect to /?room=<id>.
    role = 'display';
    roomId = normalizeRoomId(url.searchParams.get('room'));
  }

  return { role, roomId };
}

function rejectWebSocket(ws, reason) {
  try {
    ws.send(JSON.stringify({ type: 'error', error: reason }));
  } catch {
    // The connection may already be closing.
  }
  try {
    ws.close(1008, reason);
  } catch {
    // Nothing else to do.
  }
}

// Ask the Python server to turn text into a .pose binary. Returns an
// ArrayBuffer, or null on failure (caller should skip the broadcast).
async function generatePose(text) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), POSE_TIMEOUT_MS);
  try {
    const res = await fetch(`${POSE_SERVER_URL}/pose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (!res.ok) {
      console.error(`[Pose] Python server error: ${res.status}`);
      return null;
    }
    return await res.arrayBuffer();
  } catch (err) {
    const message = err.name === 'AbortError'
      ? `Timed out after ${POSE_TIMEOUT_MS}ms`
      : err.message;
    console.error('[Pose] Failed to reach Python server:', message);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

function broadcastToRoom(room, data) {
  room.lastActivityAt = Date.now();
  for (const client of room.displayClients) {
    if (client.readyState === 1) {
      client.send(data);
    }
  }
}

function broadcastError(room, message, scope = 'service') {
  broadcastToRoom(room, JSON.stringify({
    type: 'error',
    roomId: room.id,
    scope,
    text: message,
  }));
}

function resetRoomPlayback(room) {
  room.poseGeneration += 1;
  for (const session of room.audioSessions) session.reset();
  broadcastToRoom(room, JSON.stringify({ type: 'reset', roomId: room.id }));
  console.log(`[Room ${room.id}] Reset playback and cleared pending generation ${room.poseGeneration}.`);
}

wss.on('connection', (ws, request) => {
  const { role, roomId } = parseWebSocketRequest(request);
  if (!roomId) {
    rejectWebSocket(ws, 'A valid room id is required.');
    return;
  }

  const room = getRoom(roomId);

  if (role === 'display') {
    room.displayClients.add(ws);
    console.log(`[Room ${room.id}] Display client connected (${room.displayClients.size} display client(s)).`);
    ws.on('close', () => {
      room.displayClients.delete(ws);
      console.log(`[Room ${room.id}] Display client disconnected (${room.displayClients.size} display client(s)).`);
      releaseRoom(room);
    });
    ws.on('message', (message) => {
      try {
        const msg = JSON.parse(message.toString());
        if (msg.type === 'reset') resetRoomPlayback(room);
      } catch {
        // Display clients only send small JSON control messages.
      }
    });
    return;
  }

  console.log(`[Room ${room.id}] Audio client connected — opening Deepgram live session.`);
  if (!deepgram) {
    const message = 'Speech recognition is not configured. Set DEEPGRAM_API_KEY to use Speak to Sign.';
    broadcastError(room, message, 'speech');
    rejectWebSocket(ws, message);
    releaseRoom(room);
    return;
  }

  // Sentence-level batching: ASL gloss reordering needs a whole clause, so we
  // accumulate finalized words and flush a full utterance at a time — on
  // speech_final (end of utterance), at a safety cap (so a long monologue still
  // signs), and on disconnect. Each utterance is glossed to ASL then signed.
  const MAX_BUFFERED_WORDS = 16;
  let wordBuffer = [];
  let finalFlushTimer = null;
  // Serialize gloss+pose generation per connection so clips are broadcast in order.
  let poseChain = Promise.resolve();

  const session = {
    reset() {
      wordBuffer = [];
      if (finalFlushTimer) {
        clearTimeout(finalFlushTimer);
        finalFlushTimer = null;
      }
      poseChain = Promise.resolve();
    },
  };
  room.audioSessions.add(session);

  function enqueuePose(text) {
    const generation = room.poseGeneration;
    poseChain = poseChain
      .then(async () => {
        const signableText = expandDigitsForSigning(text);
        const gloss = expandDigitsForSigning(await englishToAslGloss(signableText));
        if (gloss !== text) console.log(`[Gloss] "${text}" -> "${gloss}"`);
        const arrayBuffer = await generatePose(gloss);
        if (arrayBuffer && generation === room.poseGeneration) {
          broadcastToRoom(room, Buffer.from(arrayBuffer));
        } else if (generation === room.poseGeneration) {
          broadcastError(room, 'Signing service is unavailable. Make sure the Python pose server is running.', 'pose');
        }
      })
      .catch((err) => console.error('[Pose] generation error:', err));
  }

  function flushUtterance() {
    if (finalFlushTimer) {
      clearTimeout(finalFlushTimer);
      finalFlushTimer = null;
    }
    if (wordBuffer.length > 0) {
      enqueuePose(wordBuffer.splice(0).join(' '));
    }
  }

  function scheduleFinalFlush() {
    if (finalFlushTimer || FINAL_FLUSH_DELAY_MS < 0) return;
    finalFlushTimer = setTimeout(flushUtterance, FINAL_FLUSH_DELAY_MS);
  }

  let dgLive;
  try {
    dgLive = deepgram.listen.live({
      model: 'nova-3',
      language: 'en-US',
      interim_results: true,
      smart_format: true,
      encoding: 'linear16',
      sample_rate: 16000,
      channels: 1,
      endpointing: 300,
    });

    dgLive.on(LiveTranscriptionEvents.Open, () => {
      console.log('Deepgram connection open.');
    });

    dgLive.on(LiveTranscriptionEvents.Transcript, (data) => {
      const transcript = data?.channel?.alternatives?.[0]?.transcript;
      if (!transcript) return;

      // Always push the transcript text for the on-screen display.
      broadcastToRoom(room, JSON.stringify({
        type: 'transcript',
        roomId: room.id,
        text: transcript,
        is_final: data.is_final,
      }));

      // Only finalized words feed the pose pipeline (interim words can change).
      if (!data.is_final) return;

      const signableTranscript = expandDigitsForSigning(transcript);
      const words = signableTranscript.trim().split(/\s+/).filter(Boolean);
      if (words.length) wordBuffer.push(...words);

      // Flush immediately at end-of-speech or when the buffer gets long enough.
      // Otherwise, flush shortly after finalized text so the avatar starts
      // signing before Deepgram's endpoint detector notices a full pause.
      if (data.speech_final || wordBuffer.length >= MAX_BUFFERED_WORDS) {
        flushUtterance();
      } else {
        scheduleFinalFlush();
      }
    });

    dgLive.on(LiveTranscriptionEvents.Error, (err) => {
      console.error('Deepgram error:', err);
      broadcastError(room, 'Speech recognition error. Check Deepgram credentials and network access.', 'speech');
    });

    dgLive.on(LiveTranscriptionEvents.Close, () => {
      console.log('Deepgram connection closed.');
    });
  } catch (err) {
    console.error('Failed to open Deepgram live session:', err);
  }

  if (!dgLive) {
    const message = 'Could not open the speech recognition connection.';
    broadcastError(room, message, 'speech');
    room.audioSessions.delete(session);
    rejectWebSocket(ws, message);
    releaseRoom(room);
    return;
  }

  ws.on('message', (chunk) => {
    try {
      if (dgLive && dgLive.getReadyState() === 1) {
        dgLive.send(chunk);
      }
    } catch (err) {
      console.error('Error forwarding audio to Deepgram:', err);
    }
  });

  ws.on('close', () => {
    room.audioSessions.delete(session);
    console.log(`[Room ${room.id}] Audio client disconnected — finishing Deepgram session.`);
    // Sign whatever words are left over so the final utterance isn't dropped.
    flushUtterance();
    try {
      if (dgLive) dgLive.finish();
    } catch (err) {
      console.error('Error finishing Deepgram session:', err);
    }
    releaseRoom(room);
  });

  ws.on('error', (err) => {
    console.error('Audio WebSocket error:', err);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`ASL Pose MVP server listening on http://localhost:${PORT}`);
  console.log(`Make sure the Python pose server is running first:`);
  console.log(`  cd python && uvicorn server:app --port 8000`);
  console.log(`Speaker page: / redirects to /speak?room=<room-id>`);
  console.log(`Glasses page: /glasses?room=<same-room-id>`);
});
