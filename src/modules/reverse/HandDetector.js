import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

// MediaPipe hand skeleton connections (index pairs)
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]

export class HandDetector {
  constructor() {
    this._landmarker = null
    this._ready = false
    this._lastTimestamp = -1
  }

  async init() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    )

    const makeOptions = delegate => ({
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate,
      },
      runningMode: 'VIDEO',
      numHands: 2,
    })

    try {
      this._landmarker = await HandLandmarker.createFromOptions(vision, makeOptions('GPU'))
    } catch {
      this._landmarker = await HandLandmarker.createFromOptions(vision, makeOptions('CPU'))
    }

    this._ready = true
  }

  get isReady() {
    return this._ready
  }

  detect(videoEl, timestampMs) {
    if (!this._ready) return null
    if (timestampMs === this._lastTimestamp) return null
    this._lastTimestamp = timestampMs
    return this._landmarker.detectForVideo(videoEl, timestampMs)
  }

  // Returns Float32Array(126) — 21 landmarks × 3 × 2 hands, zeros for absent hand
  flattenLandmarks(results) {
    const flat = new Float32Array(21 * 3 * 2)
    if (!results?.landmarks) return flat
    const [lh, rh] = results.landmarks
    if (lh) lh.forEach((p, i) => { flat[i*3]=p.x; flat[i*3+1]=p.y; flat[i*3+2]=p.z })
    if (rh) rh.forEach((p, i) => { flat[63+i*3]=p.x; flat[63+i*3+1]=p.y; flat[63+i*3+2]=p.z })
    return flat
  }

  // Draw hand skeleton on canvas. Flips x to match mirrored webcam display.
  drawLandmarks(canvas, results) {
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!results?.landmarks?.length) return

    const w = canvas.width
    const h = canvas.height

    for (const hand of results.landmarks) {
      // Connections
      ctx.strokeStyle = 'rgba(99,102,241,0.75)'
      ctx.lineWidth = 2
      for (const [a, b] of CONNECTIONS) {
        const pa = hand[a], pb = hand[b]
        ctx.beginPath()
        ctx.moveTo((1 - pa.x) * w, pa.y * h)
        ctx.lineTo((1 - pb.x) * w, pb.y * h)
        ctx.stroke()
      }

      // Keypoints
      for (let i = 0; i < hand.length; i++) {
        const p = hand[i]
        ctx.beginPath()
        ctx.arc((1 - p.x) * w, p.y * h, i === 0 ? 6 : 4, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? 'rgba(236,72,153,0.9)' : 'rgba(139,92,246,0.9)'
        ctx.fill()
      }
    }
  }
}
