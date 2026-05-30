import { CONFIG } from '../config.js';

export class SignPlayer {
  constructor() {
    this._currentSignId = null;
    this._signLabel = '';
    this._startTime = 0;
    this._duration = 1400; // placeholder animation duration ms

    // Hidden video element for real .mp4 sign assets
    this._video = document.createElement('video');
    this._video.playsInline = true;
    this._video.muted = true;
    this._video.style.cssText = 'position:absolute;left:-9999px;top:-9999px;';
    document.body.appendChild(this._video);

    this._mode = 'idle'; // 'video' | 'placeholder' | 'idle'
  }

  get isActive() {
    return this._currentSignId !== null;
  }

  async play(signId) {
    this._currentSignId = signId;
    this._signLabel = signId.replace(/^sign_/, '').replace(/_/g, ' ').toUpperCase();

    if (CONFIG.DEBUG_MODE) console.log('[SignPlayer] Playing:', signId);

    try {
      await this._playVideo(signId);
    } catch {
      await this._playPlaceholder();
    }

    this._mode = 'idle';
    this._currentSignId = null;
  }

  async _playVideo(signId) {
    return new Promise((resolve, reject) => {
      this._video.src = `animations/${signId}.mp4`;
      this._mode = 'video';

      const cleanup = () => {
        this._video.onerror = null;
        this._video.onended = null;
        this._video.oncanplaythrough = null;
      };

      this._video.onerror = () => { cleanup(); reject(new Error('no video')); };
      this._video.onended = () => { cleanup(); resolve(); };
      this._video.load();
      this._video.play().catch(err => { cleanup(); reject(err); });
    });
  }

  async _playPlaceholder() {
    this._mode = 'placeholder';
    this._startTime = performance.now();
    await new Promise(r => setTimeout(r, this._duration));
  }

  /** Called every animation frame by AROverlay */
  render(ctx, x, y, w, h) {
    if (!this._currentSignId) return;

    if (this._mode === 'video' && this._video.readyState >= 2) {
      this._renderVideo(ctx, x, y, w, h);
    } else {
      const t = Math.min(1, (performance.now() - this._startTime) / this._duration);
      this._renderPlaceholder(ctx, x, y, w, h, t);
    }
  }

  _renderVideo(ctx, x, y, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 16);
    ctx.clip();
    ctx.drawImage(this._video, x, y, w, h);
    ctx.restore();
  }

  _renderPlaceholder(ctx, x, y, w, h, t) {
    const cx = x + w / 2;
    const cy = y + h / 2;
    const r = Math.min(w, h) * 0.46;

    ctx.save();

    // Outer glow
    const pulse = 0.94 + 0.06 * Math.sin(t * Math.PI * 8);
    const glow = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 1.3);
    glow.addColorStop(0, 'rgba(29,158,117,0.0)');
    glow.addColorStop(1, 'rgba(29,158,117,0.18)');
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.3 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    // Background disc
    ctx.beginPath();
    ctx.arc(cx, cy, r * pulse, 0, Math.PI * 2);
    const bg = ctx.createRadialGradient(cx, cy - r * 0.2, 0, cx, cy, r * pulse);
    bg.addColorStop(0, 'rgba(29,158,117,0.92)');
    bg.addColorStop(1, 'rgba(8,48,36,0.90)');
    ctx.fillStyle = bg;
    ctx.fill();

    // Clip everything inside the disc
    ctx.clip();

    const sc = h / 300;

    // Head
    ctx.fillStyle = '#f5c18a';
    ctx.beginPath();
    ctx.arc(cx, y + h * 0.26, 26 * sc, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = '#3a2010';
    ctx.beginPath();
    ctx.arc(cx - 8 * sc, y + h * 0.24, 3 * sc, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + 8 * sc, y + h * 0.24, 3 * sc, 0, Math.PI * 2);
    ctx.fill();

    // Body
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 11 * sc;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, y + h * 0.36);
    ctx.lineTo(cx, y + h * 0.60);
    ctx.stroke();

    // Animated arms — alternating signing motion
    const swing  = Math.sin(t * Math.PI * 7);
    const swing2 = Math.cos(t * Math.PI * 5 + 0.8);
    const shouldY = y + h * 0.42;

    // Left arm
    const lx = cx - (52 + swing * 22) * sc;
    const ly = shouldY + (20 - swing2 * 28) * sc;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 10 * sc;
    ctx.beginPath();
    ctx.moveTo(cx - 10 * sc, shouldY);
    ctx.lineTo(lx, ly);
    ctx.stroke();
    ctx.fillStyle = '#f5c18a';
    ctx.beginPath();
    ctx.arc(lx, ly, 12 * sc, 0, Math.PI * 2);
    ctx.fill();

    // Right arm
    const rx = cx + (52 - swing * 18) * sc;
    const ry = shouldY + (-12 + swing2 * 32) * sc;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 10 * sc;
    ctx.beginPath();
    ctx.moveTo(cx + 10 * sc, shouldY);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.fillStyle = '#f5c18a';
    ctx.beginPath();
    ctx.arc(rx, ry, 12 * sc, 0, Math.PI * 2);
    ctx.fill();

    // Sign label background pill (inside clip)
    const label = this._signLabel;
    const fs = Math.max(11, Math.min(18, Math.round(w * 0.085)));
    ctx.font = `bold ${fs}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const tw = ctx.measureText(label).width;
    const pillH = fs + 12;
    const pillY = cy + r * 0.64;

    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.beginPath();
    ctx.roundRect(cx - tw / 2 - 10, pillY - pillH / 2, tw + 20, pillH, pillH / 2);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.fillText(label, cx, pillY);

    ctx.restore();
  }

  stop() {
    this._video.pause();
    this._video.src = '';
    this._currentSignId = null;
    this._mode = 'idle';
  }

  destroy() {
    this.stop();
    this._video.remove();
  }
}
