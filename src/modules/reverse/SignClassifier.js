// Sends frame buffer to the Python TGCN server. Returns gloss prediction or null.
export class SignClassifier {
  constructor(endpoint, confidenceThreshold) {
    this.endpoint = endpoint
    this.threshold = confidenceThreshold
    this._busy = false
  }

  async classify(frames) {
    if (this._busy) return null
    this._busy = true
    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
        signal: AbortSignal.timeout(5000),
      })
      const data = await res.json()
      if (data.error) return null
      if (data.confidence >= this.threshold) {
        return { gloss: data.gloss, confidence: data.confidence, top3: data.top3 }
      }
      return null
    } catch {
      return null
    } finally {
      this._busy = false
    }
  }
}
