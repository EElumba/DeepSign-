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
import QRCode from 'qrcode';

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
const PAIRING_LINK_TTL_MS = Number(process.env.PAIRING_LINK_TTL_MS || 30 * 60 * 1000);
const ROOM_ID_PATTERN = /^[a-zA-Z0-9_-]{6,80}$/;
const PAIRING_TOKEN_PATTERN = /^[a-zA-Z0-9_-]{16,128}$/;
const DEMO_POSE_ENABLED = process.env.DEMO_POSE_ENABLED !== '0';
const ALLOW_CUSTOM_DEMO_POSE = process.env.ALLOW_CUSTOM_DEMO_POSE === '1';

const DEMO_PHRASES = Object.freeze([
  {
    id: 'hello',
    category: 'Greeting',
    label: 'Good morning',
    text: 'good morning nice meet you',
  },
  {
    id: 'help',
    category: 'Help',
    label: 'Ask for help',
    text: 'please help me',
  },
  {
    id: 'emergency',
    category: 'Emergency',
    label: 'Call a doctor',
    text: 'emergency please call doctor',
  },
  {
    id: 'directions',
    category: 'Directions',
    label: 'Find cafeteria',
    text: 'where cafeteria',
  },
  {
    id: 'introduction',
    category: 'Introduction',
    label: 'My name is Alex',
    text: 'my name alex nice meet you',
  },
  {
    id: 'yes-no',
    category: 'Yes / no',
    label: 'Quick answer',
    text: 'yes no understand',
  },
  {
    id: 'repeat',
    category: 'Repeat',
    label: 'Say that again',
    text: 'please repeat again',
  },
  {
    id: 'thanks',
    category: 'Thank you',
    label: 'Thanks for helping',
    text: 'thank you for help',
  },
  {
    id: 'school-office',
    category: 'School desk',
    label: 'Office check-in',
    text: 'please help student find school office',
  },
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
app.set('trust proxy', true);
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

function normalizePairingToken(value) {
  const token = String(value || '').trim();
  return PAIRING_TOKEN_PATTERN.test(token) ? token : '';
}

function roomPath(pathname, roomId) {
  return `${pathname}?room=${encodeURIComponent(roomId)}`;
}

function requestOrigin(req) {
  const forwardedProto = String(req.get('x-forwarded-proto') || '').split(',')[0].trim();
  const forwardedHost = String(req.get('x-forwarded-host') || '').split(',')[0].trim();
  const proto = forwardedProto || req.protocol || 'http';
  const host = forwardedHost || req.get('host') || `localhost:${PORT}`;
  return `${proto}://${host}`;
}

function absoluteRoomUrl(req, pathname, roomId) {
  const url = new URL(pathname, requestOrigin(req));
  url.searchParams.set('room', roomId);
  return url.toString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function sendRoomLinkError(res, status, title, message) {
  res.status(status).type('html').send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #000; color: #fff; font-family: monospace; padding: 20px;
    }
    main {
      width: min(100%, 440px); border: 1px solid #333; border-radius: 8px;
      background: #111; padding: 18px; display: grid; gap: 12px;
    }
    h1 { margin: 0; font-size: 20px; }
    p { margin: 0; color: #ddd; line-height: 1.45; }
    a {
      display: inline-block; color: #000; background: #ffe14d;
      padding: 10px 12px; border-radius: 6px; text-decoration: none;
      font-weight: 700; text-align: center;
    }
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(title)}</h1>
    <p>${escapeHtml(message)}</p>
    <a href="/speak">Start a new conversation</a>
  </main>
</body>
</html>`);
}

function ensurePairingToken(room) {
  if (!room.pairingToken || Date.now() >= room.pairingExpiresAt) {
    room.pairingToken = randomUUID().replace(/-/g, '');
    room.pairingExpiresAt = Date.now() + PAIRING_LINK_TTL_MS;
  }
  return room.pairingToken;
}

function pairingQrPath(room, token) {
  return `/api/sessions/${encodeURIComponent(room.id)}/glasses-qr.svg?pair=${encodeURIComponent(token)}`;
}

function pairingGlassesUrl(req, room, token) {
  const url = new URL('/glasses', requestOrigin(req));
  url.searchParams.set('room', room.id);
  url.searchParams.set('pair', token);
  return url.toString();
}

function pairingPayload(req, room) {
  const token = ensurePairingToken(room);
  return {
    roomId: room.id,
    glassesUrl: pairingGlassesUrl(req, room, token),
    legacyGlassesUrl: absoluteRoomUrl(req, '/glasses', room.id),
    qrSvgUrl: pairingQrPath(room, token),
    expiresAt: new Date(room.pairingExpiresAt).toISOString(),
    ttlMs: Math.max(0, room.pairingExpiresAt - Date.now()),
  };
}

function validatePairingToken(room, token) {
  if (!token) return { ok: true, legacy: true };
  if (!normalizePairingToken(token)) {
    return {
      ok: false,
      status: 400,
      title: 'Invalid pairing link',
      message: 'This glasses link is malformed. Use the QR code or copy link from the main conversation screen.',
    };
  }
  if (!room) {
    return {
      ok: false,
      status: 410,
      title: 'Pairing link expired',
      message: 'This glasses pairing link is no longer active. Start a new conversation on the main device and scan a fresh QR code.',
    };
  }
  if (!room.pairingToken || token !== room.pairingToken) {
    return {
      ok: false,
      status: 403,
      title: 'Invalid pairing link',
      message: 'This glasses link does not match the active conversation. Scan the QR code again from the main device.',
    };
  }
  if (Date.now() >= room.pairingExpiresAt) {
    return {
      ok: false,
      status: 410,
      title: 'Pairing link expired',
      message: 'This glasses pairing link has expired. Refresh the QR code on the main device and scan again.',
    };
  }
  return { ok: true };
}

function roomFromHttpRequest(req) {
  return normalizeRoomId(req.query?.room);
}

function sendRoomPage(req, res, pathname, filename) {
  const hasRoomParam = Object.prototype.hasOwnProperty.call(req.query || {}, 'room');
  const roomId = roomFromHttpRequest(req);
  if (!roomId) {
    if (hasRoomParam) {
      sendRoomLinkError(
        res,
        400,
        'Invalid room link',
        'This room link is malformed. Start a fresh conversation and pair the glasses again.'
      );
      return;
    }
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
  const room = getRoom(roomId);
  const pairing = pairingPayload(req, room);
  releaseRoom(room);
  res.json({
    roomId,
    speakUrl: roomPath('/speak', roomId),
    glassesUrl: roomPath('/glasses', roomId),
    pairingGlassesUrl: pairing.glassesUrl,
    legacyGlassesUrl: pairing.legacyGlassesUrl,
    qrSvgUrl: pairing.qrSvgUrl,
    pairingExpiresAt: pairing.expiresAt,
    audioWsPath: `/ws/audio/${roomId}`,
    displayWsPath: `/ws/display/${roomId}`,
  });
});

app.get('/api/sessions/:roomId/pairing', (req, res) => {
  const roomId = normalizeRoomId(req.params.roomId);
  if (!roomId) return res.status(400).json({ error: 'A valid room id is required.' });

  const room = getRoom(roomId);
  const pairing = pairingPayload(req, room);
  releaseRoom(room);
  res.json(pairing);
});

app.get('/api/sessions/:roomId/glasses-qr.svg', async (req, res) => {
  const roomId = normalizeRoomId(req.params.roomId);
  if (!roomId) return res.status(400).type('text/plain').send('Invalid room id');

  const token = normalizePairingToken(req.query?.pair);
  if (!token) return res.status(400).type('text/plain').send('Invalid pairing token');
  const room = rooms.get(roomId);
  const validation = validatePairingToken(room, token);
  if (!validation.ok) {
    return res.status(validation.status).type('text/plain').send(validation.title);
  }

  try {
    const svg = await QRCode.toString(pairingGlassesUrl(req, room, token), {
      type: 'svg',
      margin: 2,
      width: 260,
      color: {
        dark: '#000000',
        light: '#ffffff',
      },
    });
    res
      .status(200)
      .type('image/svg+xml')
      .set('Cache-Control', 'no-store')
      .send(svg);
  } catch (err) {
    console.error('[Pairing] Failed to render QR:', err.message);
    res.status(500).type('text/plain').send('QR unavailable');
  }
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
  const hasRoomParam = Object.prototype.hasOwnProperty.call(req.query || {}, 'room');
  const hasPairParam = Object.prototype.hasOwnProperty.call(req.query || {}, 'pair');
  const roomId = roomFromHttpRequest(req);
  const token = normalizePairingToken(req.query?.pair);

  if (hasPairParam && !hasRoomParam) {
    sendRoomLinkError(
      res,
      400,
      'Invalid pairing link',
      'This glasses pairing link is missing its room. Start a fresh conversation and scan the QR code again.'
    );
    return;
  }

  if (hasPairParam && !token) {
    sendRoomLinkError(
      res,
      400,
      'Invalid pairing link',
      'This glasses pairing link is malformed. Use the QR code or copy link from the main conversation screen.'
    );
    return;
  }

  if (hasRoomParam && roomId && hasPairParam) {
    const validation = validatePairingToken(rooms.get(roomId), token);
    if (!validation.ok) {
      sendRoomLinkError(res, validation.status, validation.title, validation.message);
      return;
    }
  }

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
  let clientType = 'display';

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

  if (role === 'display') {
    const requestedType = String(url.searchParams.get('client') || '').trim().toLowerCase();
    if (requestedType === 'glasses') clientType = 'glasses';
    else if (requestedType === 'controller' || requestedType === 'main' || requestedType === 'speak') {
      clientType = 'controller';
    }
  }

  return { role, roomId, clientType };
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

function displayClientCount(room, clientType) {
  let count = 0;
  for (const client of room.displayClients) {
    if (client.readyState === 1 && client.clientType === clientType) count++;
  }
  return count;
}

function broadcastPairingState(room) {
  const glassesCount = displayClientCount(room, 'glasses');
  broadcastToRoom(room, JSON.stringify({
    type: 'pairing',
    roomId: room.id,
    glassesConnected: glassesCount > 0,
    glassesCount,
    displayClients: room.displayClients.size,
  }));
}

function resetRoomPlayback(room) {
  room.poseGeneration += 1;
  for (const session of room.audioSessions) session.reset();
  broadcastToRoom(room, JSON.stringify({ type: 'reset', roomId: room.id }));
  console.log(`[Room ${room.id}] Reset playback and cleared pending generation ${room.poseGeneration}.`);
}

wss.on('connection', (ws, request) => {
  const { role, roomId, clientType } = parseWebSocketRequest(request);
  if (!roomId) {
    rejectWebSocket(ws, 'A valid room id is required.');
    return;
  }

  const room = getRoom(roomId);

  if (role === 'display') {
    ws.clientType = clientType;
    room.displayClients.add(ws);
    console.log(`[Room ${room.id}] Display client connected as ${clientType} (${room.displayClients.size} display client(s)).`);
    broadcastPairingState(room);
    ws.on('close', () => {
      room.displayClients.delete(ws);
      console.log(`[Room ${room.id}] Display client disconnected (${room.displayClients.size} display client(s)).`);
      broadcastPairingState(room);
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
