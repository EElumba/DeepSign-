import { CONFIG } from '../config.js';
import { lookup } from './SignDictionary.js';

export class WordQueue {
  constructor({ signPlayer, onSignStart, onSignComplete, onGap }) {
    this._signPlayer = signPlayer;
    this._onSignStart = onSignStart;
    this._onSignComplete = onSignComplete;
    this._onGap = onGap;

    this._queue = [];
    this._isPlaying = false;
    this._stats = { total: 0, matched: 0, gaps: 0 };
  }

  enqueue(transcript) {
    const normalized = transcript.toLowerCase().replace(/[^\w\s]/g, '').trim();
    const tokens = normalized.split(/\s+/).filter(Boolean);

    for (const word of tokens) {
      this._stats.total++;
      const signId = lookup(word);
      if (signId) {
        this._queue.push({ word, signId });
        this._stats.matched++;
      } else {
        this._onGap(word);
        this._stats.gaps++;
      }
    }

    if (!this._isPlaying && this._queue.length > 0) {
      this._isPlaying = true;
      this._drain();
    }
  }

  getStats() {
    return { ...this._stats };
  }

  get isPlaying() {
    return this._isPlaying;
  }

  async _drain() {
    while (this._queue.length > 0) {
      const { word, signId } = this._queue.shift();

      this._onSignStart(word, signId);
      await this._signPlayer.play(signId);
      this._onSignComplete(word, signId);

      if (this._queue.length > 0) {
        await this._sleep(CONFIG.SIGN_DELAY_MS);
      }
    }
    this._isPlaying = false;
  }

  _sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }
}
