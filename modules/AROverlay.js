import { CONFIG } from '../config.js';

export class AROverlay {
  constructor(container, signPlayer) {
    this._container = container;
    this._signPlayer = signPlayer;

    // --- Phase 1.1 ---
    // Camera feed — CSS handles positioning & mirroring
    this._video = document.createElement('video');
    this._video.autoplay = true;
    this._video.playsInline = true;
    this._video.muted = true;
    container.appendChild(this._video);

    // Compositing canvas — transparent except avatar + captions
    this._canvas = document.createElement('canvas');
    this._ctx = this._canvas.getContext('2d');
    container.appendChild(this._canvas);

    this._rafId = null;
    this._finalWords = [];   // normalised words from confirmed transcripts
    this._interimText = '';  // provisional interim transcript
    this._currentWord = null; // word currently being signed (for highlighting)
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: CONFIG.CAMERA_FACING,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    this._video.srcObject = stream;
    await new Promise(r => { this._video.onloadedmetadata = r; });
    await this._video.play();

    this._resize();
    window.addEventListener('resize', () => this._resize());
    this._loop();
  }

  _resize() {
    this._canvas.width = this._container.offsetWidth;
    this._canvas.height = this._container.offsetHeight;
  }

  // Called by main when a final transcript arrives
  addFinalTranscript(text) {
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').trim().split(/\s+/).filter(Boolean);
    this._finalWords.push(...words);
    if (this._finalWords.length > 20) this._finalWords = this._finalWords.slice(-20);
    this._interimText = '';
  }

  // Called by main with rolling interim text
  setInterim(text) {
    this._interimText = text;
  }

  // Called by WordQueue callbacks
  setCurrentWord(word) {
    this._currentWord = word;
  }

  _loop() {
    this._rafId = requestAnimationFrame(() => this._loop());
    this._render();
  }

  _render() {
    const ctx = this._ctx;
    const W = this._canvas.width;
    const H = this._canvas.height;
    if (!W || !H) return;

    ctx.clearRect(0, 0, W, H);

    // --- Phase 1.4 ---
    // Sign avatar — bottom-right
    if (this._signPlayer.isActive) {
      const size = Math.round(W * CONFIG.AVATAR_SCALE);
      const x = W - size - 20;
      const y = H - size - 70;
      this._signPlayer.render(ctx, x, y, size, size);
    }

    // Caption bar — bottom-centre
    if (CONFIG.SHOW_CAPTIONS) this._drawCaptions(ctx, W, H);
  }

  _drawCaptions(ctx, W, H) {
    const BAR_H = 58;
    const BAR_Y = H - BAR_H;

    ctx.fillStyle = 'rgba(0,0,0,0.62)';
    ctx.fillRect(0, BAR_Y, W, BAR_H);

    const interimWords = this._interimText
      ? this._interimText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean)
      : [];

    const allWords = [
      ...this._finalWords.map(w => ({ text: w, kind: 'final' })),
      ...interimWords.map(w => ({ text: w, kind: 'interim' })),
    ];

    if (allWords.length === 0) {
      // Idle hint
      ctx.fillStyle = 'rgba(255,255,255,0.22)';
      ctx.font = '15px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Listening for speech…', W / 2, BAR_Y + BAR_H / 2);
      return;
    }

    ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
    ctx.textBaseline = 'middle';
    const CY = BAR_Y + BAR_H / 2;
    const SPACE_W = ctx.measureText(' ').width;
    const PAD = 24;

    const widths = allWords.map(({ text }) => ctx.measureText(text).width);
    let totalW = widths.reduce((a, b) => a + b, 0) + SPACE_W * (allWords.length - 1);

    // Scroll so latest words are always visible
    const MAX_W = W - PAD * 2;
    let startX = W / 2 - totalW / 2;
    if (totalW > MAX_W) startX = W - PAD - totalW;

    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD, BAR_Y, W - PAD * 2, BAR_H);
    ctx.clip();

    let x = startX;
    for (let i = 0; i < allWords.length; i++) {
      const { text, kind } = allWords[i];
      if (kind === 'interim') {
        ctx.fillStyle = 'rgba(255,255,255,0.40)';
      } else if (text === this._currentWord) {
        ctx.fillStyle = CONFIG.CAPTION_HIGHLIGHT_COLOR;
      } else {
        ctx.fillStyle = '#fff';
      }
      ctx.textAlign = 'left';
      ctx.fillText(text, x, CY);
      x += widths[i] + SPACE_W;
    }

    ctx.restore();
  }

  stop() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._video.srcObject?.getTracks().forEach(t => t.stop());
  }
}
