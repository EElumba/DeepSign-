import React, { useRef, useState, useEffect, useCallback } from 'react'
import WebcamFeed from './components/WebcamFeed'
import GestureDisplay from './components/GestureDisplay'
import { initGestureRecognizer, recognizeGesture } from './utils/mediapipe'

// How long (ms) the user must hold a sign before it's appended to the sentence
const HOLD_MS = 1200
const MIN_CONFIDENCE = 0.72

export default function App() {
  const webcamRef = useRef(null)
  const rafRef = useRef(null)
  const holdRef = useRef({ gesture: null, since: null, committed: false })

  const [isReady, setIsReady] = useState(false)
  const [gesture, setGesture] = useState('')
  const [confidence, setConfidence] = useState(0)
  const [sentence, setSentence] = useState('')

  // Initialize MediaPipe on mount
  useEffect(() => {
    initGestureRecognizer()
      .then(() => setIsReady(true))
      .catch(err => console.error('MediaPipe init failed:', err))
  }, [])

  const processFrame = useCallback(() => {
    const video = webcamRef.current?.video
    if (video && video.readyState >= 2) {
      const results = recognizeGesture(video, performance.now())
      const top = results?.gestures?.[0]?.[0]

      if (top && top.score >= MIN_CONFIDENCE && top.categoryName !== 'None') {
        const name = top.categoryName
        setGesture(name)
        setConfidence(top.score)

        const hold = holdRef.current
        if (hold.gesture === name) {
          if (!hold.committed && performance.now() - hold.since >= HOLD_MS) {
            setSentence(prev => prev + name)
            hold.committed = true
          }
        } else {
          holdRef.current = { gesture: name, since: performance.now(), committed: false }
        }
      } else {
        setGesture('')
        setConfidence(0)
        holdRef.current = { gesture: null, since: null, committed: false }
      }
    }

    rafRef.current = requestAnimationFrame(processFrame)
  }, [])

  useEffect(() => {
    if (!isReady) return
    rafRef.current = requestAnimationFrame(processFrame)
    return () => cancelAnimationFrame(rafRef.current)
  }, [isReady, processFrame])

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      <header className="py-6 text-center">
        <h1 className="text-3xl font-black tracking-tight">
          Deep<span className="text-indigo-400">Sign</span>
        </h1>
        <p className="text-gray-500 text-sm mt-1">Real-time sign language translator — runs entirely in your browser</p>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 pb-8 grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        <WebcamFeed ref={webcamRef} isReady={isReady} />
        <GestureDisplay
          gesture={gesture}
          confidence={confidence}
          sentence={sentence}
          onBackspace={() => setSentence(s => s.slice(0, -1))}
          onClear={() => setSentence('')}
          onSpace={() => setSentence(s => s + ' ')}
        />
      </main>
    </div>
  )
}
