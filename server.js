import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fetch from 'node-fetch';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT || 3000);
const POSE_SERVER_URL = process.env.POSE_SERVER_URL || 'http://localhost:8000';
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY || '';
const deepgram = DEEPGRAM_API_KEY ? createClient(DEEPGRAM_API_KEY) : null;

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', async (req, res) => {
  const python = await checkPythonHealth();
  res.json({
    status: 'ok',
    deepgram: Boolean(DEEPGRAM_API_KEY),
    python,
  });
});

app.post('/api/sign', async (req, res) => {
  const text = String(req.body?.text || '').trim();
  if (!text) {
    res.status(400).json({ error: 'text must be non-empty' });
    return;
  }

  const envelope = await createMotionEnvelope(text, {
    source: 'typed',
    speechFinal: true,
    isFinal: true,
  });
  broadcastToDisplays(envelope);
  res.json(envelope);
});

const server = createServer(app);
const wss = new WebSocketServer({ server });
const displayClients = new Set();

async function checkPythonHealth() {
  try {
    const res = await fetch(`${POSE_SERVER_URL}/health`, { timeout: 2000 });
    if (!res.ok) return { ok: false, status: res.status };
    return { ok: true, ...(await res.json()) };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function createMotionEnvelope(text, meta = {}) {
  try {
    const res = await fetch(`${POSE_SERVER_URL}/motion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        generate_pose: true,
        source: meta.source || 'speech',
      }),
    });

    if (!res.ok) {
      const detail = await safeText(res);
      throw new Error(`Python motion server returned ${res.status}: ${detail}`);
    }

    return {
      type: 'motion',
      id: cryptoRandomId(),
      receivedAt: Date.now(),
      transcript: {
        text,
        isFinal: Boolean(meta.isFinal),
        speechFinal: Boolean(meta.speechFinal),
        source: meta.source || 'speech',
      },
      ...(await res.json()),
    };
  } catch (err) {
    console.error('[Motion] Falling back to caption-only plan:', err.message);
    return {
      type: 'motion',
      id: cryptoRandomId(),
      receivedAt: Date.now(),
      transcript: {
        text,
        isFinal: Boolean(meta.isFinal),
        speechFinal: Boolean(meta.speechFinal),
        source: meta.source || 'speech',
      },
      plan: {
        mode: 'caption_fallback',
        sourceText: text,
        units: [{ type: 'caption', text, reason: 'motion_server_unavailable' }],
      },
      clips: [],
      warnings: [err.message],
    };
  }
}

async function safeText(res) {
  try {
    return await res.text();
  } catch {
    return '';
  }
}

function cryptoRandomId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function broadcastToDisplays(payload) {
  const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
  for (const client of displayClients) {
    if (client.readyState === 1) client.send(data);
  }
}

class TranscriptScheduler {
  constructor(onFlush) {
    this.onFlush = onFlush;
    this.words = [];
    this.maxWords = Number(process.env.MAX_WORDS_PER_SIGN_JOB || 8);
  }

  observe(data) {
    const alt = data?.channel?.alternatives?.[0];
    const transcript = String(alt?.transcript || '').trim();
    if (!transcript) return;

    broadcastToDisplays({
      type: 'transcript',
      text: transcript,
      isFinal: Boolean(data.is_final),
      speechFinal: Boolean(data.speech_final),
    });

    if (!data.is_final) return;

    const words = transcript.split(/\s+/).filter(Boolean);
    this.words.push(...words);

    const shouldFlush = Boolean(data.speech_final) || /[.!?]$/.test(transcript);
    this.flushReady(shouldFlush);
  }

  flushReady(flushRemainder = false) {
    while (this.words.length >= this.maxWords) {
      this.onFlush(this.words.splice(0, this.maxWords).join(' '), false);
    }
    if (flushRemainder && this.words.length) {
      this.onFlush(this.words.splice(0).join(' '), true);
    }
  }
}

wss.on('connection', (ws, request) => {
  const isAudio = (request.url || '/').startsWith('/audio');

  if (!isAudio) {
    displayClients.add(ws);
    ws.send(JSON.stringify({
      type: 'system',
      status: 'connected',
      deepgram: Boolean(DEEPGRAM_API_KEY),
      poseServerUrl: POSE_SERVER_URL,
    }));
    console.log(`Display client connected (${displayClients.size} total).`);
    ws.on('close', () => {
      displayClients.delete(ws);
      console.log(`Display client disconnected (${displayClients.size} total).`);
    });
    return;
  }

  if (!deepgram) {
    ws.send(JSON.stringify({ type: 'error', message: 'DEEPGRAM_API_KEY is not configured.' }));
    ws.close();
    return;
  }

  console.log('Audio client connected; opening Deepgram live session.');
  let motionChain = Promise.resolve();
  const scheduler = new TranscriptScheduler((text, speechFinal) => {
    motionChain = motionChain
      .then(async () => {
        const envelope = await createMotionEnvelope(text, {
          source: 'speech',
          isFinal: true,
          speechFinal,
        });
        broadcastToDisplays(envelope);
      })
      .catch((err) => console.error('[Motion] queue error:', err));
  });

  let dgLive;
  try {
    dgLive = deepgram.listen.live({
      model: process.env.DEEPGRAM_MODEL || 'nova-3',
      language: process.env.DEEPGRAM_LANGUAGE || 'en-US',
      interim_results: true,
      smart_format: true,
      encoding: 'linear16',
      sample_rate: 16000,
      channels: 1,
      endpointing: Number(process.env.DEEPGRAM_ENDPOINTING_MS || 300),
    });

    dgLive.on(LiveTranscriptionEvents.Open, () => {
      console.log('Deepgram connection open.');
      broadcastToDisplays({ type: 'system', status: 'speech-open' });
    });
    dgLive.on(LiveTranscriptionEvents.Transcript, (data) => scheduler.observe(data));
    dgLive.on(LiveTranscriptionEvents.Error, (err) => {
      console.error('Deepgram error:', err);
      broadcastToDisplays({ type: 'system', status: 'speech-error', message: String(err?.message || err) });
    });
    dgLive.on(LiveTranscriptionEvents.Close, () => {
      console.log('Deepgram connection closed.');
      broadcastToDisplays({ type: 'system', status: 'speech-closed' });
    });
  } catch (err) {
    console.error('Failed to open Deepgram live session:', err);
    ws.close();
    return;
  }

  ws.on('message', (chunk) => {
    try {
      if (dgLive && dgLive.getReadyState() === 1) dgLive.send(chunk);
    } catch (err) {
      console.error('Error forwarding audio to Deepgram:', err);
    }
  });

  ws.on('close', () => {
    console.log('Audio client disconnected; finishing Deepgram session.');
    scheduler.flushReady(true);
    try {
      if (dgLive) dgLive.finish();
    } catch (err) {
      console.error('Error finishing Deepgram session:', err);
    }
  });

  ws.on('error', (err) => {
    console.error('Audio WebSocket error:', err);
  });
});

server.listen(PORT, () => {
  console.log(`DeepSign server listening on http://localhost:${PORT}`);
  console.log(`Python motion server: ${POSE_SERVER_URL}`);
  if (!DEEPGRAM_API_KEY) {
    console.log('DEEPGRAM_API_KEY is not set; typed-text mode is available, mic streaming is disabled.');
  }
});
