// SnowflakeLogger — disabled for now (post-MVP feature)
// To re-enable: uncomment this file, restore imports in main.js, and populate
// SNOWFLAKE_ACCOUNT / SNOWFLAKE_JWT in config.js.

/*
import { CONFIG } from '../config.js';

export class SnowflakeLogger {
  constructor(sessionId) {
    this._sessionId = sessionId;
    this._buffer = [];
    this._timer = null;
    this._enabled = !!(CONFIG.SNOWFLAKE_ACCOUNT && CONFIG.SNOWFLAKE_JWT);

    if (this._enabled) {
      this._timer = setInterval(() => this.flush(), CONFIG.LOG_FLUSH_INTERVAL_MS);
    }
  }

  logSessionStart() {
    this._push({
      event: 'session_start',
      session_id: this._sessionId,
      timestamp: new Date().toISOString(),
      user_agent: navigator.userAgent,
    });
  }

  logSignPlayed(word, signId, sttLatencyMs = 0, animLatencyMs = 0) {
    this._push({
      event: 'sign_played',
      session_id: this._sessionId,
      word,
      sign_id: signId,
      stt_latency_ms: sttLatencyMs,
      anim_latency_ms: animLatencyMs,
      matched: true,
      timestamp: new Date().toISOString(),
    });
  }

  logGap(word) {
    this._push({
      event: 'sign_gap',
      session_id: this._sessionId,
      word,
      matched: false,
      timestamp: new Date().toISOString(),
    });
  }

  logSessionEnd(durationMs, stats = {}) {
    this._push({
      event: 'session_end',
      session_id: this._sessionId,
      duration_ms: durationMs,
      total_words: stats.total ?? 0,
      matched_words: stats.matched ?? 0,
      gap_words: stats.gaps ?? 0,
      timestamp: new Date().toISOString(),
    });
    this.flush();
  }

  _push(event) {
    this._buffer.push(event);
    if (this._buffer.length >= CONFIG.LOG_BATCH_SIZE) this.flush();
  }

  async flush() {
    if (!this._enabled || this._buffer.length === 0) return;
    const events = this._buffer.splice(0);
    try {
      await fetch(
        `https://${CONFIG.SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/statements`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${CONFIG.SNOWFLAKE_JWT}`,
            'Content-Type': 'application/json',
            'X-Snowflake-Authorization-Token-Type': 'KEYPAIR_JWT',
          },
          body: JSON.stringify({
            statement: this._buildSQL(events),
            database: CONFIG.SNOWFLAKE_DATABASE,
            schema: CONFIG.SNOWFLAKE_SCHEMA,
            warehouse: CONFIG.SNOWFLAKE_WAREHOUSE,
          }),
        }
      );
    } catch (e) {
      console.warn('Snowflake log failed (non-blocking):', e.message);
    }
  }

  _buildSQL(events) {
    const esc = v => (v == null ? 'NULL' : `'${String(v).replace(/'/g, "''")}'`);
    const rows = events.map(e =>
      `(${esc(e.event)},${esc(e.session_id)},${esc(e.word ?? null)},${esc(e.sign_id ?? null)},` +
      `${e.stt_latency_ms ?? 'NULL'},${e.anim_latency_ms ?? 'NULL'},` +
      `${e.matched != null ? e.matched : 'NULL'},${esc(e.timestamp)})`
    );
    return (
      `INSERT INTO SIGN_EVENTS ` +
      `(event_type,session_id,word,sign_id,stt_latency_ms,anim_latency_ms,matched,timestamp) ` +
      `VALUES ${rows.join(',')}`
    );
  }

  destroy() {
    if (this._timer) clearInterval(this._timer);
  }
}
*/
