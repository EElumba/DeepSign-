#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { WebSocket } from 'ws';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const pythonDir = join(repoRoot, 'python');

const startedProcesses = [];
const failures = [];

const config = {
  appPort: numberEnv('SMOKE_APP_PORT', 3130),
  posePort: numberEnv('SMOKE_POSE_PORT', 8130),
  startTimeoutMs: numberEnv('SMOKE_START_TIMEOUT_MS', 90000),
  eventTimeoutMs: numberEnv('SMOKE_EVENT_TIMEOUT_MS', 60000),
  poseTimeoutMs: numberEnv('SMOKE_POSE_TIMEOUT_MS', 60000),
  phraseId: process.env.SMOKE_PHRASE_ID || 'hello',
  reuseServices: process.env.SMOKE_REUSE_SERVICES === '1',
};

const appUrl = trimTrailingSlash(process.env.SMOKE_APP_URL || `http://127.0.0.1:${config.appPort}`);
const poseUrl = trimTrailingSlash(process.env.SMOKE_POSE_URL || `http://127.0.0.1:${config.posePort}`);

main().catch(async (error) => {
  fail('smoke test failed', error.message || String(error));
  await cleanup();
  printSummaryAndExit();
});

async function main() {
  printHeader();

  const poseWasOnline = await isJsonOk(`${poseUrl}/health`);
  if (poseWasOnline && config.reuseServices) {
    pass('python pose server is already online', poseUrl);
  } else {
    const poseProcess = startPoseServer();
    await waitForCheck('python pose server is online', async () => {
      const health = await getJson(`${poseUrl}/health`);
      assertEqual(health.status, 'ok', 'pose /health status');
      return health;
    }, config.startTimeoutMs, poseProcess);
  }

  const appWasOnline = await isJsonOk(`${appUrl}/api/health`);
  if (appWasOnline && config.reuseServices) {
    pass('frontend app is already online', appUrl);
  } else {
    const appProcess = startFrontend();
    await waitForCheck('frontend app is online', async () => {
      const health = await getJson(`${appUrl}/api/health`);
      assertEqual(health.status, 'ok', 'app /api/health status');
      return health;
    }, config.startTimeoutMs, appProcess);
  }

  const initialHealth = await getJson(`${appUrl}/api/health`);
  assertEqual(initialHealth.status, 'ok', 'app health status');
  assertEqual(initialHealth.services?.pose?.reachable, true, 'app health pose reachable');
  assertEqual(initialHealth.services?.pose?.status, 'ok', 'app health pose status');
  assertEqual(initialHealth.services?.demo?.status, 'available', 'app health demo status');
  assertEqual(initialHealth.privacy?.rawAudioStored, false, 'app health raw audio privacy');
  assertEqual(initialHealth.privacy?.rawVideoStored, false, 'app health raw video privacy');
  assertEqual(initialHealth.privacy?.fullConversationsStoredByDefault, false, 'app health conversation privacy');
  assertEqual(typeof initialHealth.metrics?.timings?.pose_generation?.count, 'number', 'app health metrics pose bucket');
  pass('health endpoints report real service state', summarizeHealth(initialHealth));

  const poseHealth = await getJson(`${poseUrl}/health`);
  assertEqual(poseHealth.status, 'ok', 'pose health status');
  pass('python /health reports ok', JSON.stringify(poseHealth));

  const session = await getJson(`${appUrl}/api/sessions/new`);
  assertMatch(session.roomId, /^[a-zA-Z0-9_-]{6,80}$/, 'room id');
  pass('created private room', session.roomId);

  await expectPageOk(`${appUrl}${session.speakUrl}`, 'speak page', 'Demo phrases');
  await expectPageOk(`${appUrl}${session.glassesUrl}`, 'glasses page', 'Animated signing avatar');
  pass('joined room pages', `${session.speakUrl} and ${session.glassesUrl}`);

  const displaySocket = await connectDisplaySocket(session.roomId);
  try {
    await waitForEvent(displaySocket, 'pairing state for glasses display', (event) => (
      event.kind === 'json'
      && event.data.type === 'pairing'
      && event.data.glassesConnected === true
      && event.data.glassesCount >= 1
    ), config.eventTimeoutMs);
    pass('glasses display websocket joined room', `/ws/display/${session.roomId}?client=glasses`);

    const pendingTranscript = waitForEvent(displaySocket, 'demo transcript on glasses display', (event) => (
      event.kind === 'json'
      && event.data.type === 'transcript'
      && event.data.source === 'demo'
      && event.data.is_final === true
    ), config.eventTimeoutMs);

    const pendingPose = waitForEvent(displaySocket, 'binary .pose on glasses display', (event) => (
      event.kind === 'binary' && event.bytes > 0
    ), config.eventTimeoutMs);
    const pendingPoseMetric = waitForEvent(displaySocket, 'pose generation metric on glasses display', (event) => (
      event.kind === 'json'
      && event.data.type === 'metrics'
      && event.data.event?.kind === 'timing'
      && event.data.event?.stage === 'pose_generation'
      && event.data.event?.status === 'ok'
      && Number.isFinite(Number(event.data.event?.durationMs))
    ), config.eventTimeoutMs);

    const demoResponse = await postJson(`${appUrl}/api/demo/pose`, {
      roomId: session.roomId,
      phraseId: config.phraseId,
    }, config.poseTimeoutMs);
    assertEqual(demoResponse.ok, true, 'demo pose response ok');
    assertEqual(demoResponse.roomId, session.roomId, 'demo pose response room');
    assertEqual(demoResponse.phraseId, config.phraseId, 'demo pose response phrase');
    assertEqual(demoResponse.discarded, false, 'demo pose response discarded');
    assertAtLeast(demoResponse.displayClients, 1, 'demo pose display clients');
    pass('sent demo phrase through /speak demo pipeline', `${demoResponse.text} -> ${demoResponse.gloss}`);

    const transcriptEvent = await pendingTranscript;
    const poseEvent = await pendingPose;
    const poseMetricEvent = await pendingPoseMetric;
    pass('glasses display received transcript', transcriptEvent.data.text);
    pass('glasses display received avatar/sign output', `${poseEvent.bytes} bytes`);
    pass('glasses display received pose timing metric', `${poseMetricEvent.data.event.durationMs}ms`);

    displaySocket.send(JSON.stringify({
      type: 'metrics',
      event: 'avatar_playback_start',
      durationMs: 12,
      pipelineId: poseMetricEvent.data.event.pipelineId,
      status: 'ok',
    }));

    const finalHealth = await waitForCheck('room metrics include demo timings', async () => {
      const health = await getJson(`${appUrl}/api/health?room=${encodeURIComponent(session.roomId)}`);
      assertAtLeast(health.room?.metrics?.timings?.gloss?.count, 1, 'room gloss timing count');
      assertAtLeast(health.room?.metrics?.timings?.pose_generation?.count, 1, 'room pose timing count');
      assertAtLeast(health.room?.metrics?.timings?.avatar_playback_start?.count, 1, 'room avatar start timing count');
      return health;
    }, 10000);
    assertEqual(finalHealth.services?.pose?.reachable, true, 'final app health pose reachable');
    assertEqual(finalHealth.services?.pose?.status, 'ok', 'final app health pose status');
    assertEqual(finalHealth.services?.demo?.status, 'available', 'final app health demo status');
    assertAtLeast(finalHealth.rooms?.active, 1, 'final active rooms');
    assertEqual(finalHealth.room?.clients?.glassesConnected, true, 'final room glasses connected');
    pass('post-run health/status is consistent', summarizeHealth(finalHealth));
  } finally {
    displaySocket.close();
  }

  await cleanup();
  printSummaryAndExit();
}

function startPoseServer() {
  const python = resolvePythonCommand();
  const child = spawn(python, [
    '-m',
    'uvicorn',
    'server:app',
    '--host',
    '127.0.0.1',
    '--port',
    String(config.posePort),
  ], {
    cwd: pythonDir,
    env: {
      ...process.env,
      PORT: String(config.posePort),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const tracked = trackProcess(child, 'pose');
  pass('starting python pose server', `${python} -m uvicorn server:app --port ${config.posePort}`);
  return tracked;
}

function startFrontend() {
  const child = spawn(process.execPath, ['server.js'], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PORT: String(config.appPort),
      POSE_SERVER_URL: poseUrl,
      OPENAI_API_KEY: '',
      ELEVENLABS_API_KEY: '',
      DEMO_POSE_ENABLED: '1',
      ROOM_IDLE_TTL_MS: '10000',
      POSE_TIMEOUT_MS: String(config.poseTimeoutMs),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const tracked = trackProcess(child, 'app');
  pass('starting frontend app', `${process.execPath} server.js on ${appUrl}`);
  return tracked;
}

function trackProcess(child, label) {
  const tracked = {
    child,
    label,
    tail: [],
    exitInfo: null,
    ready: false,
    stopping: false,
  };

  const tail = [];
  const collect = (streamName, chunk) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      tail.push(`${streamName}: ${line}`);
      while (tail.length > 20) tail.shift();
    }
  };

  child.stdout.on('data', (chunk) => collect('stdout', chunk));
  child.stderr.on('data', (chunk) => collect('stderr', chunk));
  child.on('exit', (code, signal) => {
    tracked.exitInfo = { code, signal };
    if (tracked.stopping || !tracked.ready) return;
    if (code !== null && code !== 0) {
      fail(`${label} process exited early`, `code ${code}\n${tail.join('\n')}`);
    } else if (signal && signal !== 'SIGTERM') {
      fail(`${label} process exited early`, `signal ${signal}\n${tail.join('\n')}`);
    }
  });
  child.on('error', (error) => fail(`${label} process failed to start`, error.message));

  tracked.tail = tail;
  startedProcesses.push(tracked);
  return tracked;
}

function resolvePythonCommand() {
  const configured = process.env.SMOKE_PYTHON;
  if (configured) return configured;

  const candidates = [
    join(pythonDir, '.venv', 'bin', 'python'),
    join(pythonDir, '.venv-b312', 'bin', 'python'),
    'python3',
    'python',
  ];

  return candidates.find((candidate) => !candidate.includes('/') || existsSync(candidate)) || 'python3';
}

async function connectDisplaySocket(roomId) {
  const wsUrl = toWebSocketUrl(`${appUrl}/ws/display/${encodeURIComponent(roomId)}?client=glasses`);
  const events = [];
  const socket = new WebSocket(wsUrl);
  socket.events = events;

  socket.on('message', (message, isBinary) => {
    if (isBinary) {
      events.push({ kind: 'binary', bytes: byteLength(message) });
      return;
    }

    const text = message.toString();
    try {
      events.push({ kind: 'json', data: JSON.parse(text), text });
    } catch {
      events.push({ kind: 'text', text });
    }
  });
  socket.on('close', (code, reason) => {
    events.push({ kind: 'close', code, reason: reason.toString() });
  });
  socket.on('error', (error) => {
    events.push({ kind: 'error', error });
  });

  await new Promise((resolveOpen, rejectOpen) => {
    const timer = setTimeout(() => rejectOpen(new Error(`timed out opening ${wsUrl}`)), 10000);
    socket.once('open', () => {
      clearTimeout(timer);
      resolveOpen();
    });
    socket.once('error', (error) => {
      clearTimeout(timer);
      rejectOpen(error);
    });
  });

  return socket;
}

async function waitForEvent(socket, label, predicate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let offset = 0;

  while (Date.now() < deadline) {
    while (offset < socket.events.length) {
      const event = socket.events[offset++];
      if (event.kind === 'json' && event.data.type === 'error') {
        throw new Error(`${label}: websocket error ${event.data.text || event.data.error}`);
      }
      if (event.kind === 'close') {
        throw new Error(`${label}: websocket closed code=${event.code} reason=${event.reason}`);
      }
      if (event.kind === 'error') {
        throw new Error(`${label}: websocket error ${event.error.message}`);
      }
      if (predicate(event)) return event;
    }
    await delay(100);
  }

  throw new Error(`${label}: timed out after ${timeoutMs}ms`);
}

async function waitForCheck(label, check, timeoutMs, trackedProcess = null) {
  const deadline = Date.now() + timeoutMs;
  let lastError;

  while (Date.now() < deadline) {
    try {
      const result = await check();
      if (trackedProcess) trackedProcess.ready = true;
      pass(label, 'ready');
      return result;
    } catch (error) {
      lastError = error;
      if (trackedProcess?.exitInfo) {
        const { code, signal } = trackedProcess.exitInfo;
        throw new Error(
          `${label}: ${trackedProcess.label} process exited before readiness `
          + `(code=${code ?? 'null'}, signal=${signal ?? 'null'})\n${trackedProcess.tail.join('\n')}`
        );
      }
      await delay(500);
    }
  }

  throw new Error(`${label}: timed out after ${timeoutMs}ms (${lastError?.message || 'no response'})`);
}

async function expectPageOk(url, label, expectedText) {
  const response = await fetch(url, { redirect: 'manual' });
  if (!response.ok) {
    throw new Error(`${label} returned HTTP ${response.status}`);
  }
  const body = await response.text();
  if (!body.includes(expectedText)) {
    throw new Error(`${label} did not include expected marker: ${expectedText}`);
  }
}

async function getJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`${url} returned HTTP ${response.status}: ${text}`);
  }
  return response.json();
}

async function postJson(url, body, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(`${url} returned HTTP ${response.status}: ${text}`);
    }
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

async function isJsonOk(url) {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) return false;
    await response.json();
    return true;
  } catch {
    return false;
  }
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertAtLeast(actual, expected, label) {
  if (typeof actual !== 'number' || actual < expected) {
    throw new Error(`${label}: expected at least ${expected}, got ${JSON.stringify(actual)}`);
  }
}

function assertMatch(actual, pattern, label) {
  if (!pattern.test(String(actual || ''))) {
    throw new Error(`${label}: value did not match ${pattern}: ${JSON.stringify(actual)}`);
  }
}

function byteLength(message) {
  if (typeof message === 'string') return Buffer.byteLength(message);
  if (Buffer.isBuffer(message)) return message.length;
  if (message instanceof ArrayBuffer) return message.byteLength;
  if (Array.isArray(message)) return message.reduce((sum, part) => sum + byteLength(part), 0);
  return Number(message?.byteLength || message?.length || 0);
}

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function trimTrailingSlash(value) {
  return String(value).replace(/\/+$/, '');
}

function toWebSocketUrl(url) {
  const parsed = new URL(url);
  parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
  return parsed.toString();
}

function summarizeHealth(health) {
  return [
    `pose=${health.services?.pose?.status}`,
    `poseReachable=${health.services?.pose?.reachable}`,
    `demo=${health.services?.demo?.status}`,
    `rooms=${health.rooms?.active}`,
  ].join(' ');
}

async function cleanup() {
  for (const tracked of startedProcesses.reverse()) {
    const { child, label } = tracked;
    if (child.exitCode !== null || child.signalCode !== null) continue;
    tracked.stopping = true;
    child.kill('SIGTERM');
    const exited = await Promise.race([
      new Promise((resolveExit) => child.once('exit', () => resolveExit(true))),
      delay(5000).then(() => false),
    ]);
    if (!exited) {
      child.kill('SIGKILL');
      await new Promise((resolveExit) => child.once('exit', resolveExit));
    }
    pass(`stopped ${label} process`, 'clean shutdown');
  }
}

function printHeader() {
  console.log('DeepSign real-service smoke test');
  console.log(`app:  ${appUrl}`);
  console.log(`pose: ${poseUrl}`);
  console.log('storage: no raw audio or video is captured');
  console.log('');
}

function pass(label, detail) {
  console.log(`[pass] ${label}${detail ? ` - ${detail}` : ''}`);
}

function fail(label, detail) {
  failures.push({ label, detail });
  console.error(`[fail] ${label}${detail ? ` - ${detail}` : ''}`);
}

function printSummaryAndExit() {
  if (failures.length === 0) {
    console.log('');
    console.log('Smoke test passed.');
    process.exit(0);
  }

  console.error('');
  console.error(`Smoke test failed with ${failures.length} issue(s).`);
  for (const issue of failures) {
    console.error(`- ${issue.label}${issue.detail ? `: ${issue.detail}` : ''}`);
  }
  console.error('');
  console.error('Setup hints:');
  console.error('- Install Python dependencies with: cd python && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt');
  console.error('- Reuse already-running services with: SMOKE_REUSE_SERVICES=1 SMOKE_APP_URL=http://127.0.0.1:3000 SMOKE_POSE_URL=http://127.0.0.1:8000 npm run smoke:e2e');
  process.exit(1);
}
