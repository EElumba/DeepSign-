import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fetch from 'node-fetch';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

const __dirname = dirname(fileURLToPath(import.meta.url));

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
if (!DEEPGRAM_API_KEY) {
  console.error('Missing DEEPGRAM_API_KEY. Copy .env.example to .env and fill it in.');
  process.exit(1);
}

const POSE_SERVER_URL = process.env.POSE_SERVER_URL || 'http://localhost:8000';

const deepgram = createClient(DEEPGRAM_API_KEY);

const app = express();
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'));
});

const server = createServer(app);
const wss = new WebSocketServer({ server });

// Separate connection sets so the audio handler knows who to broadcast to.
const audioClients = new Set();
const displayClients = new Set();

// Ask the Python server to turn text into a .pose binary. Returns an
// ArrayBuffer, or null on failure (caller should skip the broadcast).
async function generatePose(text) {
  try {
    const res = await fetch(`${POSE_SERVER_URL}/pose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      console.error(`[Pose] Python server error: ${res.status}`);
      return null;
    }
    return await res.arrayBuffer();
  } catch (err) {
    console.error('[Pose] Failed to reach Python server:', err.message);
    return null;
  }
}

function broadcastToDisplays(data) {
  for (const client of displayClients) {
    if (client.readyState === 1) {
      client.send(data);
    }
  }
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
    return;
  }

  audioClients.add(ws);
  console.log('Audio client connected — opening Deepgram live session.');

  // Word batching: collect finalized words and emit a pose for every WORDS_PER_CHUNK
  // words. Leftover words are carried forward and flushed at the end of each
  // utterance (speech_final) and when the connection closes — so a full sentence
  // gets signed in order instead of being cut off by the next transcript.
  const WORDS_PER_CHUNK = 4;
  let wordBuffer = [];
  // Serialize pose generation per connection so clips are broadcast in order.
  let poseChain = Promise.resolve();

  function enqueuePose(text) {
    poseChain = poseChain
      .then(async () => {
        const arrayBuffer = await generatePose(text);
        if (arrayBuffer) broadcastToDisplays(Buffer.from(arrayBuffer));
      })
      .catch((err) => console.error('[Pose] generation error:', err));
  }

  function drainChunks(flushRemainder) {
    while (wordBuffer.length >= WORDS_PER_CHUNK) {
      enqueuePose(wordBuffer.splice(0, WORDS_PER_CHUNK).join(' '));
    }
    if (flushRemainder && wordBuffer.length > 0) {
      enqueuePose(wordBuffer.splice(0).join(' '));
    }
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

      const words = transcript.trim().split(/\s+/).filter(Boolean);
      if (words.length) wordBuffer.push(...words);

      // Emit full 4-word chunks now; flush the tail at the end of an utterance.
      drainChunks(Boolean(data.speech_final));
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
    console.log('Audio client disconnected — finishing Deepgram session.');
    // Sign whatever words are left over so the final partial chunk isn't dropped.
    drainChunks(true);
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
