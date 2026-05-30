const socketDot = document.getElementById('socketDot');
const statusText = document.getElementById('statusText');
const modeBadge = document.getElementById('modeBadge');
const transcriptEl = document.getElementById('transcript');
const planEl = document.getElementById('plan');
const captionEl = document.getElementById('caption');
const poseViewer = document.getElementById('poseViewer');
const micBtn = document.getElementById('micBtn');
const textForm = document.getElementById('textForm');
const textInput = document.getElementById('textInput');

let currentPoseUrl = null;
let clipQueue = [];
let playing = false;
let watchdog = null;
let recording = false;
let audioContext = null;
let micStream = null;
let micSocket = null;
let micSource = null;
let workletNode = null;
const seenMotionIds = new Set();

function setStatus(kind, text) {
  socketDot.className = `dot ${kind || ''}`.trim();
  statusText.textContent = text;
}

function renderPlan(plan) {
  if (!plan?.units?.length) {
    planEl.textContent = 'No sign units available.';
    return;
  }
  planEl.textContent = plan.units.map((unit) => {
    if (unit.type === 'sign') return `SIGN ${unit.gloss}`;
    if (unit.type === 'fingerspell') return `FS ${unit.letters.join('-')}`;
    return `CAPTION ${unit.text}`;
  }).join('\n');
}

function base64ToBlob(base64, mime) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime || 'application/octet-stream' });
}

function enqueuePoseClip(blob) {
  clipQueue.push(URL.createObjectURL(blob));
  if (!playing) playNextClip();
}

function playNextClip() {
  if (watchdog) clearTimeout(watchdog);
  watchdog = null;

  if (currentPoseUrl) {
    URL.revokeObjectURL(currentPoseUrl);
    currentPoseUrl = null;
  }

  if (clipQueue.length === 0) {
    playing = false;
    try {
      if (typeof poseViewer.pause === 'function') poseViewer.pause();
      poseViewer.removeAttribute('src');
    } catch {}
    return;
  }

  playing = true;
  currentPoseUrl = clipQueue.shift();
  const apply = () => {
    poseViewer.src = currentPoseUrl;
    try {
      if (typeof poseViewer.play === 'function') poseViewer.play();
    } catch {}
    watchdog = setTimeout(playNextClip, 30000);
  };

  if (customElements?.whenDefined) {
    customElements.whenDefined('pose-viewer').then(apply).catch(apply);
  } else {
    apply();
  }
}

poseViewer.addEventListener('ended$', playNextClip);

function handleMotion(envelope) {
  if (envelope.id) {
    if (seenMotionIds.has(envelope.id)) return;
    seenMotionIds.add(envelope.id);
    if (seenMotionIds.size > 100) {
      const first = seenMotionIds.values().next().value;
      seenMotionIds.delete(first);
    }
  }

  transcriptEl.textContent = envelope.transcript?.text || envelope.plan?.sourceText || '';
  captionEl.textContent = envelope.transcript?.text || '';
  renderPlan(envelope.plan);

  const readyPoseClips = (envelope.clips || [])
    .filter((clip) => clip.kind === 'pose' && clip.status === 'ready' && clip.data);

  for (const clip of readyPoseClips) {
    enqueuePoseClip(base64ToBlob(clip.data, clip.mime));
  }
}

const displaySocket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`);
displaySocket.onopen = () => setStatus('live', 'Connected');
displaySocket.onclose = () => setStatus('error', 'Disconnected');
displaySocket.onerror = () => setStatus('error', 'Socket error');
displaySocket.onmessage = (event) => {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch {
    return;
  }

  if (message.type === 'system') {
    if (message.deepgram) modeBadge.textContent = 'Speech ready';
    if (message.status) setStatus('live', message.status);
    return;
  }

  if (message.type === 'transcript') {
    transcriptEl.textContent = message.text;
    return;
  }

  if (message.type === 'motion') {
    handleMotion(message);
  }
};

textForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = '';
  transcriptEl.textContent = text;
  captionEl.textContent = text;

  const res = await fetch('/api/sign', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    setStatus('error', 'Sign request failed');
    return;
  }
  handleMotion(await res.json());
});

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  micSocket = new WebSocket(`${protocol}://${location.host}/audio`);
  micSocket.binaryType = 'arraybuffer';
  micSocket.onclose = () => {
    if (recording) stopMic();
  };

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  audioContext = new AudioCtx();
  await audioContext.audioWorklet.addModule('/js/audio-worklet.js');

  micSource = audioContext.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioContext, 'pcm16-capture');
  workletNode.port.onmessage = (event) => {
    if (micSocket?.readyState === WebSocket.OPEN) micSocket.send(event.data);
  };
  micSource.connect(workletNode);

  recording = true;
  micBtn.textContent = 'Stop mic';
  micBtn.classList.add('recording');
  modeBadge.textContent = 'Speech mode';
}

function stopMic() {
  try { workletNode?.disconnect(); } catch {}
  try { micSource?.disconnect(); } catch {}
  try { audioContext?.close(); } catch {}
  try { micStream?.getTracks().forEach((track) => track.stop()); } catch {}
  try { if (micSocket?.readyState === WebSocket.OPEN) micSocket.close(); } catch {}

  recording = false;
  workletNode = null;
  micSource = null;
  audioContext = null;
  micStream = null;
  micSocket = null;
  micBtn.textContent = 'Start mic';
  micBtn.classList.remove('recording');
  modeBadge.textContent = 'Typed mode';
}

micBtn.addEventListener('click', async () => {
  micBtn.disabled = true;
  try {
    if (recording) stopMic();
    else await startMic();
  } catch (error) {
    console.error(error);
    setStatus('error', error.message || 'Mic failed');
    stopMic();
  } finally {
    micBtn.disabled = false;
  }
});
