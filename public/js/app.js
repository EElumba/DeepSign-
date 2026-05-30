const socketDot = document.getElementById('socketDot');
const statusText = document.getElementById('statusText');
const modeBadge = document.getElementById('modeBadge');
const transcriptEl = document.getElementById('transcript');
const planEl = document.getElementById('plan');
const captionEl = document.getElementById('caption');
const signCanvas = document.getElementById('signCanvas');
const micBtn = document.getElementById('micBtn');
const textForm = document.getElementById('textForm');
const textInput = document.getElementById('textInput');

let recording = false;
let audioContext = null;
let micStream = null;
let micSocket = null;
let micSource = null;
let workletNode = null;
const seenMotionIds = new Set();
const signer = createProceduralSigner(signCanvas);

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
    if (unit.type === 'sign') return `SIGN ${unit.gloss} ${formatHands(unit.hands)}`;
    if (unit.type === 'fingerspell') return `FS ${unit.letters.join('-')} ${formatHands(unit.hands)}`;
    return `CAPTION ${unit.text}`;
  }).join('\n');
}

function formatHands(hands) {
  if (!hands?.pattern) return '[hands: unknown]';
  const active = Array.isArray(hands.active) ? hands.active.join('+') : 'dominant';
  if (hands.pattern === 'symmetrical' && hands.alternating) {
    return `[two-hand symmetrical alternating: ${active}]`;
  }
  if (hands.pattern === 'symmetrical') {
    return `[two-hand symmetrical: ${active}]`;
  }
  if (hands.pattern === 'asymmetrical') {
    const nonDominantRole = hands.nonDominant?.role || 'support';
    return `[two-hand asymmetrical: dominant + ${nonDominantRole}]`;
  }
  return `[one-hand: ${active}]`;
}

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
  signer.enqueuePlan(envelope.plan);
}

function createProceduralSigner(canvas) {
  const ctx = canvas.getContext('2d');
  const queue = [];
  let active = null;
  let activeStarted = 0;
  let raf = null;

  const neutral = {
    dominant: { x: 0.66, y: 0.72 },
    nonDominant: { x: 0.34, y: 0.72 },
  };

  function enqueuePlan(plan) {
    if (!plan?.units?.length) return;
    queue.push(...plan.units.filter((unit) => unit.type === 'sign' || unit.type === 'fingerspell'));
    if (!raf) {
      active = null;
      activeStarted = performance.now();
      raf = requestAnimationFrame(tick);
    }
  }

  function tick(now) {
    ensureCanvasSize();
    if (!active && queue.length) {
      active = queue.shift();
      activeStarted = now;
    }

    const duration = active ? durationForUnit(active) : 900;
    const progress = active ? Math.min(1, (now - activeStarted) / duration) : 0;
    draw(active, progress, now);

    if (active && progress >= 1) {
      active = null;
      activeStarted = now;
    }

    if (active || queue.length) {
      raf = requestAnimationFrame(tick);
    } else {
      raf = requestAnimationFrame((nextNow) => {
        ensureCanvasSize();
        draw(null, 0, nextNow);
        raf = null;
      });
    }
  }

  function durationForUnit(unit) {
    if (unit.type === 'fingerspell') return Math.max(900, unit.letters.length * 210);
    if (unit.hands?.pattern === 'symmetrical' && unit.hands?.alternating) return 1150;
    if (unit.hands?.pattern === 'symmetrical') return 950;
    if (unit.hands?.pattern === 'asymmetrical') return 1050;
    return 800;
  }

  function ensureCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function draw(unit, progress, now) {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const body = skeletonPoints(w, h);
    const hands = handTargets(unit, progress, now);
    const dominant = point(hands.dominant, w, h);
    const nonDominant = point(hands.nonDominant, w, h);
    const dominantElbow = elbowPoint(body.rightShoulder, dominant, 1);
    const nonDominantElbow = elbowPoint(body.leftShoulder, nonDominant, -1);

    drawGuide(w, h);
    drawBody(body);
    drawArm(body.leftShoulder, nonDominantElbow, nonDominant, '#78d9ff');
    drawArm(body.rightShoulder, dominantElbow, dominant, '#52d273');
    drawHand(nonDominant, '#78d9ff', unit, 'nonDominant', progress);
    drawHand(dominant, '#52d273', unit, 'dominant', progress);
    drawLabel(unit, w, h);
  }

  function skeletonPoints(w, h) {
    return {
      head: { x: w * 0.5, y: h * 0.2 },
      neck: { x: w * 0.5, y: h * 0.31 },
      chest: { x: w * 0.5, y: h * 0.46 },
      hips: { x: w * 0.5, y: h * 0.68 },
      leftShoulder: { x: w * 0.38, y: h * 0.34 },
      rightShoulder: { x: w * 0.62, y: h * 0.34 },
    };
  }

  function handTargets(unit, progress, now) {
    if (!unit) {
      return {
        dominant: idle(neutral.dominant, now, 0),
        nonDominant: idle(neutral.nonDominant, now, 1),
      };
    }

    const eased = easeInOut(progress);
    const pulse = Math.sin(progress * Math.PI * 2);
    const hands = unit.hands || { pattern: 'one_handed' };

    if (unit.type === 'fingerspell') {
      const letterIndex = Math.min(unit.letters.length - 1, Math.floor(progress * unit.letters.length));
      const offset = (letterIndex % 3 - 1) * 0.025;
      return {
        dominant: mix(neutral.dominant, { x: 0.58 + offset, y: 0.43 + Math.abs(pulse) * 0.025 }, eased),
        nonDominant: mix(neutral.nonDominant, { x: 0.42, y: 0.48 }, eased),
      };
    }

    if (hands.pattern === 'symmetrical' && hands.alternating) {
      return {
        dominant: mix(neutral.dominant, { x: 0.61, y: 0.48 + pulse * 0.09 }, eased),
        nonDominant: mix(neutral.nonDominant, { x: 0.39, y: 0.48 - pulse * 0.09 }, eased),
      };
    }

    if (hands.pattern === 'symmetrical') {
      const y = 0.5 - Math.abs(pulse) * 0.07;
      return {
        dominant: mix(neutral.dominant, { x: 0.61, y }, eased),
        nonDominant: mix(neutral.nonDominant, { x: 0.39, y }, eased),
      };
    }

    if (hands.pattern === 'asymmetrical') {
      const target = hands.nonDominant?.role === 'target'
        ? { x: 0.43, y: 0.48 }
        : { x: 0.45, y: 0.55 };
      return {
        dominant: mix(neutral.dominant, { x: 0.56 + Math.abs(pulse) * 0.04, y: 0.48 - Math.abs(pulse) * 0.1 }, eased),
        nonDominant: mix(neutral.nonDominant, target, eased),
      };
    }

    return {
      dominant: mix(neutral.dominant, { x: 0.58, y: 0.45 - Math.abs(pulse) * 0.05 }, eased),
      nonDominant: mix(neutral.nonDominant, { x: 0.39, y: 0.62 }, eased * 0.45),
    };
  }

  function drawGuide(w, h) {
    ctx.save();
    ctx.strokeStyle = 'rgba(244, 241, 232, 0.08)';
    ctx.lineWidth = 1;
    for (const y of [0.34, 0.5, 0.66]) {
      ctx.beginPath();
      ctx.moveTo(w * 0.18, h * y);
      ctx.lineTo(w * 0.82, h * y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawBody(body) {
    ctx.save();
    ctx.strokeStyle = '#f4f1e8';
    ctx.fillStyle = '#f4f1e8';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    line(body.neck, body.chest);
    line(body.chest, body.hips);
    line(body.leftShoulder, body.rightShoulder);
    ctx.beginPath();
    ctx.arc(body.head.x, body.head.y, 34, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawArm(shoulder, elbow, hand, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    line(shoulder, elbow);
    line(elbow, hand);
    ctx.restore();
  }

  function drawHand(hand, color, unit, side, progress) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = '#061008';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(hand.x, hand.y, side === 'dominant' ? 17 : 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const shape = handShape(unit, side, progress);
    ctx.fillStyle = '#061008';
    ctx.font = '700 12px ui-sans-serif, system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(shape, hand.x, hand.y + 0.5);
    ctx.restore();
  }

  function drawLabel(unit, w, h) {
    if (!unit) return;
    const label = unit.type === 'fingerspell'
      ? `FS ${unit.letters[Math.min(unit.letters.length - 1, Math.floor(unit.letters.length * 0.999))] || ''}`
      : unit.gloss;
    ctx.save();
    ctx.fillStyle = '#f4f1e8';
    ctx.font = '700 20px ui-sans-serif, system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(label, w * 0.5, h * 0.9);
    ctx.restore();
  }

  function handShape(unit, side, progress) {
    if (!unit) return '';
    if (unit.type === 'fingerspell') {
      const index = Math.min(unit.letters.length - 1, Math.floor(progress * unit.letters.length));
      return side === 'dominant' ? unit.letters[index] : 'B';
    }
    if (side === 'nonDominant' && unit.hands?.pattern === 'asymmetrical') return 'B';
    if (unit.hands?.pattern === 'symmetrical') return 'S';
    return side === 'dominant' ? '1' : 'B';
  }

  function line(a, b) {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function elbowPoint(shoulder, hand, bendDirection) {
    const mid = { x: (shoulder.x + hand.x) / 2, y: (shoulder.y + hand.y) / 2 };
    const dx = hand.x - shoulder.x;
    const dy = hand.y - shoulder.y;
    const length = Math.max(1, Math.hypot(dx, dy));
    return {
      x: mid.x + (-dy / length) * 36 * bendDirection,
      y: mid.y + (dx / length) * 36 * bendDirection,
    };
  }

  function point(p, w, h) {
    return { x: p.x * w, y: p.y * h };
  }

  function mix(a, b, t) {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
    };
  }

  function idle(base, now, phase) {
    return {
      x: base.x,
      y: base.y + Math.sin(now / 700 + phase) * 0.01,
    };
  }

  function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  ensureCanvasSize();
  draw(null, 0, performance.now());
  window.addEventListener('resize', () => {
    ensureCanvasSize();
    draw(active, 0, performance.now());
  });

  return { enqueuePlan };
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
