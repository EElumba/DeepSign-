// Sliding window of the last N frames of hand landmark data.
export class FrameBuffer {
  constructor(nFrames = 30) {
    this.nFrames = nFrames
    this.buffer = []
  }

  push(landmarks) {
    this.buffer.push(Array.from(landmarks))
    if (this.buffer.length > this.nFrames) this.buffer.shift()
  }

  isFull() {
    return this.buffer.length >= this.nFrames
  }

  // Returns (N, 165) array — pads 126-float hand landmarks to TGCN's expected 165-float input
  getFrames() {
    return this.buffer.map(frame => {
      const padded = new Array(165).fill(0)
      frame.forEach((v, i) => { if (i < 165) padded[i] = v })
      return padded
    })
  }

  clear() {
    this.buffer = []
  }
}
