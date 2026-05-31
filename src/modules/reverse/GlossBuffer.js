// Accumulates confirmed gloss words. Detects signing pause to flush → sentence builder.
// onGlossAdded is called on every push/clear with the current gloss array.
export class GlossBuffer {
  constructor(pauseMs, onSentenceReady, onGlossAdded) {
    this.pauseMs = pauseMs
    this.onSentenceReady = onSentenceReady
    this.onGlossAdded = onGlossAdded
    this.glosses = []
    this._lastGloss = null
    this._flushTimer = null
  }

  push(gloss) {
    if (gloss === this._lastGloss) return
    this._lastGloss = gloss
    this.glosses.push(gloss)
    this.onGlossAdded?.([...this.glosses])

    clearTimeout(this._flushTimer)
    this._flushTimer = setTimeout(() => this._flush(), this.pauseMs)
  }

  _flush() {
    if (this.glosses.length === 0) return
    const sentence = [...this.glosses]
    this.glosses = []
    this._lastGloss = null
    this.onGlossAdded?.([])
    this.onSentenceReady(sentence)
  }

  clear() {
    clearTimeout(this._flushTimer)
    this.glosses = []
    this._lastGloss = null
    this.onGlossAdded?.([])
  }
}
