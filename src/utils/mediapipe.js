import { GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision'

let recognizer = null

export async function initGestureRecognizer() {
  if (recognizer) return recognizer

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  )

  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: '/gesture_recognizer.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  })

  return recognizer
}

export function recognizeGesture(videoEl, timestampMs) {
  if (!recognizer) return null
  return recognizer.recognizeForVideo(videoEl, timestampMs)
}
