import { CONFIG } from './config.js';
import { AudioCapture } from './modules/AudioCapture.js';
import { WordQueue } from './modules/WordQueue.js';
import { SignPlayer } from './modules/SignPlayer.js';
import { AROverlay } from './modules/AROverlay.js';
// import { SnowflakeLogger } from './logging/SnowflakeLogger.js'; // disabled — post-MVP

// ── Session ────────────────────────────────────────────────────────────────
// const sessionId = crypto.randomUUID(); // used by Snowflake logger
let sessionStartTime = null;

// ── DOM ───────────────────────────────────────────────────────────────────
const container   = document.getElementById('ar-container');
const startOverlay = document.getElementById('start-overlay');
const startBtn    = document.getElementById('start-btn');
const statusBar   = document.getElementById('status-bar');
const statusDot   = document.getElementById('status-dot');
const statusText  = document.getElementById('status-text');
const errorMsg    = document.getElementById('error-msg');

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

// ── Modules ───────────────────────────────────────────────────────────────
// const logger = new SnowflakeLogger(sessionId); // disabled — post-MVP
const signPlayer = new SignPlayer();
const arOverlay  = new AROverlay(container, signPlayer);

const wordQueue = new WordQueue({
  signPlayer,
  onSignStart: (word, signId) => {
    arOverlay.setCurrentWord(word);
    setStatus('signing', `Signing: ${word}`);
    if (CONFIG.DEBUG_MODE) console.log('[Queue] sign start →', word, signId);
  },
  onSignComplete: (word, signId) => {
    arOverlay.setCurrentWord(null);
    setStatus('listening', 'Listening…');
    // logger.logSignPlayed(word, signId); // disabled — post-MVP
  },
  onGap: word => {
    // logger.logGap(word); // disabled — post-MVP
    if (CONFIG.DEBUG_MODE) console.log('[Queue] gap (no sign):', word);
  },
});

const audioCapture = new AudioCapture({
  onFinal: text => {
    if (CONFIG.DEBUG_MODE) console.log('[STT] final:', text);
    arOverlay.addFinalTranscript(text);
    wordQueue.enqueue(text);
  },
  onInterim: text => {
    arOverlay.setInterim(text);
  },
});

// ── Start ─────────────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  errorMsg.style.display = 'none';
  statusBar.removeAttribute('hidden');
  setStatus('', 'Opening camera…');

  try {
    await arOverlay.start();
    setStatus('', 'Opening microphone…');
    await audioCapture.start();

    startOverlay.style.display = 'none';
    sessionStartTime = Date.now();
    // logger.logSessionStart(); // disabled — post-MVP
    setStatus('listening', 'Listening…');
  } catch (err) {
    console.error('[main] Start failed:', err);
    startBtn.disabled = false;
    startBtn.textContent = 'Start Live Translation';
    setStatus('error', 'Error');
    errorMsg.textContent = err.message || 'Could not access camera or microphone.';
    errorMsg.style.display = 'block';
  }
});

// ── Teardown ──────────────────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
  // Snowflake session-end logging disabled — post-MVP
  // if (sessionStartTime) {
  //   logger.logSessionEnd(Date.now() - sessionStartTime, wordQueue.getStats());
  // }
  audioCapture.stop();
  arOverlay.stop();
  signPlayer.destroy();
  // logger.destroy(); // disabled — post-MVP
});
