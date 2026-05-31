import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fetch from 'node-fetch';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';
import OpenAI from 'openai';

const __dirname = dirname(fileURLToPath(import.meta.url));

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
if (!DEEPGRAM_API_KEY) {
  console.error('Missing DEEPGRAM_API_KEY. Copy .env.example to .env and fill it in.');
  process.exit(1);
}

// Strip trailing slashes so `${POSE_SERVER_URL}/pose` never becomes `//pose`
// (a double slash 404s on FastAPI/Starlette).
const POSE_SERVER_URL = (process.env.POSE_SERVER_URL || 'http://localhost:8000').replace(/\/+$/, '');
const FINAL_FLUSH_DELAY_MS = Number(process.env.FINAL_FLUSH_DELAY_MS || 120);
const POSE_TIMEOUT_MS = Number(process.env.POSE_TIMEOUT_MS || 30000);

const deepgram = createClient(DEEPGRAM_API_KEY);

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
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'));
});

const server = createServer(app);
const wss = new WebSocketServer({ server });

// Separate connection sets so the audio handler knows who to broadcast to.
const audioClients = new Set();
const displayClients = new Set();
const audioSessions = new Set();
let poseGeneration = 0;

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

function broadcastToDisplays(data) {
  for (const client of displayClients) {
    if (client.readyState === 1) {
      client.send(data);
    }
  }
}

function resetPosePlayback() {
  poseGeneration += 1;
  for (const session of audioSessions) session.reset();
  broadcastToDisplays(JSON.stringify({ type: 'reset' }));
  console.log(`[Pose] Reset playback and cleared pending generation ${poseGeneration}.`);
}

wss.on('connection', (ws, request) => {
  const isAudio = (request.url || '/').startsWith('/audio');

  if (!isAudio) {
    displayClients.add(ws);
    console.log(`Display client connected (${displayClients.size} total).`);
    ws.on('close', () => {
      displayClients.delete(ws);
      console.log(`Display client disconnected (${displayClients.size} total).`);
    });
    ws.on('message', (message) => {
      try {
        const msg = JSON.parse(message.toString());
        if (msg.type === 'reset') resetPosePlayback();
      } catch {
        // Display clients only send small JSON control messages.
      }
    });
    return;
  }

  audioClients.add(ws);
  console.log('Audio client connected — opening Deepgram live session.');

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
  audioSessions.add(session);

  function enqueuePose(text) {
    const generation = poseGeneration;
    poseChain = poseChain
      .then(async () => {
        const signableText = expandDigitsForSigning(text);
        const gloss = expandDigitsForSigning(await englishToAslGloss(signableText));
        if (gloss !== text) console.log(`[Gloss] "${text}" -> "${gloss}"`);
        const arrayBuffer = await generatePose(gloss);
        if (arrayBuffer && generation === poseGeneration) {
          broadcastToDisplays(Buffer.from(arrayBuffer));
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
      broadcastToDisplays(JSON.stringify({
        type: 'transcript',
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
    });

    dgLive.on(LiveTranscriptionEvents.Close, () => {
      console.log('Deepgram connection closed.');
    });
  } catch (err) {
    console.error('Failed to open Deepgram live session:', err);
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
    audioClients.delete(ws);
    audioSessions.delete(session);
    console.log('Audio client disconnected — finishing Deepgram session.');
    // Sign whatever words are left over so the final utterance isn't dropped.
    flushUtterance();
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

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`ASL Pose MVP server listening on http://localhost:${PORT}`);
  console.log(`Make sure the Python pose server is running first:`);
  console.log(`  cd python && uvicorn server:app --port 8000`);
  console.log(`On the Meta glasses browser, open http://YOUR_LOCAL_IP:${PORT} and grant mic permission.`);
});
