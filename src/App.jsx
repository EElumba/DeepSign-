import React, { useRef, useState, useEffect, useCallback } from 'react'
import WebcamFeed from './components/WebcamFeed'
import GestureDisplay from './components/GestureDisplay'
import SignToVoicePanel from './components/SignToVoicePanel'
import SettingsModal from './components/SettingsModal'
import { initGestureRecognizer, recognizeGesture } from './utils/mediapipe'
import { HandDetector } from './modules/reverse/HandDetector'
import { FrameBuffer } from './modules/reverse/FrameBuffer'
import { SignClassifier } from './modules/reverse/SignClassifier'
import { GlossBuffer } from './modules/reverse/GlossBuffer'
import { SentenceBuilder } from './modules/reverse/SentenceBuilder'
import { TTSPlayer } from './modules/reverse/TTSPlayer'
import { CONFIG } from './config'

export default function App() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)
  const holdRef   = useRef({ gesture: null, since: null, committed: false })

  // Reverse pipeline — stable object refs, not state
  const reverseActiveRef    = useRef(false)
  const handDetectorRef     = useRef(null)
  const frameBufferRef      = useRef(null)
  const classifierRef       = useRef(null)
  const glossBufferRef      = useRef(null)
  const sentenceBuilderRef  = useRef(null)
  const ttsPlayerRef        = useRef(null)

  const [mode, setMode]               = useState('forward')
  const [showSettings, setShowSettings] = useState(false)

  // Forward mode state
  const [isForwardReady, setIsForwardReady] = useState(false)
  const [gesture, setGesture]     = useState('')
  const [confidence, setConfidence] = useState(0)
  const [sentence, setSentence]   = useState('')

  // Reverse mode state
  const [isReverseReady, setIsReverseReady]   = useState(false)
  const [currentSign, setCurrentSign]         = useState(null)
  const [signConfidence, setSignConfidence]   = useState(0)
  const [glosses, setGlosses]                 = useState([])
  const [currentSentence, setCurrentSentence] = useState('')
  const [isSpeaking, setIsSpeaking]           = useState(false)
  const [classifierStatus, setClassifierStatus] = useState('checking')

  // ─── Forward mode init ───────────────────────────────────────────────────────
  useEffect(() => {
    initGestureRecognizer()
      .then(() => setIsForwardReady(true))
      .catch(err => console.error('GestureRecognizer init failed:', err))
  }, [])

  const forwardLoop = useCallback(() => {
    const video = webcamRef.current?.video
    if (video && video.readyState >= 2) {
      const results = recognizeGesture(video, performance.now())
      const top = results?.gestures?.[0]?.[0]

      if (top && top.score >= CONFIG.MIN_CONFIDENCE && top.categoryName !== 'None') {
        const name = top.categoryName
        setGesture(name)
        setConfidence(top.score)

        const hold = holdRef.current
        if (hold.gesture === name) {
          if (!hold.committed && performance.now() - hold.since >= CONFIG.HOLD_MS) {
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
    rafRef.current = requestAnimationFrame(forwardLoop)
  }, [])

  useEffect(() => {
    if (!isForwardReady || mode !== 'forward') return
    rafRef.current = requestAnimationFrame(forwardLoop)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [isForwardReady, mode, forwardLoop])

  // ─── Reverse mode frame loop ─────────────────────────────────────────────────
  const reverseLoop = useCallback((timestamp) => {
    if (!reverseActiveRef.current) return

    const video    = webcamRef.current?.video
    const canvas   = canvasRef.current
    const detector = handDetectorRef.current

    if (video && video.readyState >= 2 && detector?.isReady) {
      // Keep canvas pixel dims in sync with video
      if (canvas && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
        canvas.width  = video.videoWidth  || 640
        canvas.height = video.videoHeight || 480
      }

      const results = detector.detect(video, timestamp)
      if (results) {
        if (canvas) detector.drawLandmarks(canvas, results)
        const flat = detector.flattenLandmarks(results)
        frameBufferRef.current?.push(flat)
      }
    }

    rafRef.current = requestAnimationFrame(reverseLoop)
  }, [])

  // Classifier polling — separate from RAF so the heavy fetch doesn't block rendering
  useEffect(() => {
    if (mode !== 'reverse') return
    const interval = setInterval(async () => {
      if (!reverseActiveRef.current) return
      const fb = frameBufferRef.current
      const cl = classifierRef.current
      if (!fb?.isFull() || !cl) return

      const result = await cl.classify(fb.getFrames())
      if (result && reverseActiveRef.current) {
        setCurrentSign(result.gloss)
        setSignConfidence(result.confidence)
        glossBufferRef.current?.push(result.gloss)
      }
    }, 300)
    return () => clearInterval(interval)
  }, [mode])

  // ─── Reverse mode lifecycle ───────────────────────────────────────────────────
  const startReverse = useCallback(async () => {
    setIsReverseReady(false)
    setCurrentSign(null)
    setSignConfidence(0)
    setGlosses([])
    setCurrentSentence('')
    setIsSpeaking(false)
    setClassifierStatus('checking')
    reverseActiveRef.current = true

    // Health-check the classifier
    const healthUrl = CONFIG.CLASSIFIER_ENDPOINT.replace('/classify', '/health')
    try {
      const res = await fetch(healthUrl, { signal: AbortSignal.timeout(3000) })
      setClassifierStatus(res.ok ? 'online' : 'offline')
    } catch {
      setClassifierStatus('offline')
    }

    // Initialize HandLandmarker once (survives mode switches)
    if (!handDetectorRef.current) {
      const detector = new HandDetector()
      await detector.init()
      handDetectorRef.current = detector
    }

    // Fresh pipeline objects each time reverse mode starts
    frameBufferRef.current    = new FrameBuffer(CONFIG.N_FRAMES)
    classifierRef.current     = new SignClassifier(CONFIG.CLASSIFIER_ENDPOINT, CONFIG.CONFIDENCE_THRESHOLD)
    sentenceBuilderRef.current = new SentenceBuilder()
    ttsPlayerRef.current      = new TTSPlayer()

    const sb  = sentenceBuilderRef.current
    const tts = ttsPlayerRef.current

    glossBufferRef.current = new GlossBuffer(
      CONFIG.SIGNING_PAUSE_MS,
      async (glossList) => {
        if (!reverseActiveRef.current) return
        setCurrentSentence('...')
        const built = await sb.build(glossList)
        if (!reverseActiveRef.current) return
        setCurrentSentence(built)
        setIsSpeaking(true)
        await tts.speak(built)
        if (reverseActiveRef.current) setIsSpeaking(false)
      },
      (updated) => {
        if (reverseActiveRef.current) setGlosses([...updated])
      }
    )

    setIsReverseReady(true)
    rafRef.current = requestAnimationFrame(reverseLoop)
  }, [reverseLoop])

  const stopReverse = useCallback(() => {
    reverseActiveRef.current = false
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    glossBufferRef.current?.clear()
    frameBufferRef.current?.clear()

    const canvas = canvasRef.current
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

    setIsReverseReady(false)
    setCurrentSign(null)
    setSignConfidence(0)
    setIsSpeaking(false)
  }, [])

  // ─── Mode switching ───────────────────────────────────────────────────────────
  const switchMode = useCallback((newMode) => {
    if (newMode === mode) return
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }

    if (newMode === 'forward') {
      stopReverse()
      setMode('forward')
      // forwardLoop useEffect restarts on mode change
    } else {
      setGesture('')
      setConfidence(0)
      holdRef.current = { gesture: null, since: null, committed: false }
      setMode('reverse')
      startReverse()
    }
  }, [mode, stopReverse, startReverse])

  // ─── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      <header className="py-4 px-6 flex items-center justify-between border-b border-gray-800/60 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-black tracking-tight leading-none">
            Deep<span className="text-indigo-400">Sign</span>
          </h1>
          <p className="text-gray-600 text-xs mt-0.5">Real-time ASL translator</p>
        </div>

        {/* Mode toggle */}
        <div className="flex items-center gap-1.5 bg-gray-800 rounded-full p-1">
          <button
            onClick={() => switchMode('forward')}
            className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all ${
              mode === 'forward'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            ✋ Sign → Text
          </button>
          <button
            onClick={() => switchMode('reverse')}
            className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all ${
              mode === 'reverse'
                ? 'bg-emerald-600 text-white shadow'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            🔊 Sign → Voice
          </button>
        </div>

        {/* Settings */}
        <button
          onClick={() => setShowSettings(true)}
          className="w-9 h-9 rounded-xl bg-gray-800 hover:bg-gray-700 transition-colors text-gray-400 hover:text-white flex items-center justify-center text-base"
          title="Settings"
        >
          ⚙
        </button>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 pb-8 pt-6 grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        <WebcamFeed
          ref={webcamRef}
          canvasRef={canvasRef}
          isReady={mode === 'forward' ? isForwardReady : isReverseReady}
          mode={mode}
          currentSign={currentSign}
          signConfidence={signConfidence}
        />

        {mode === 'forward' ? (
          <GestureDisplay
            gesture={gesture}
            confidence={confidence}
            sentence={sentence}
            onBackspace={() => setSentence(s => s.slice(0, -1))}
            onClear={() => setSentence('')}
            onSpace={() => setSentence(s => s + ' ')}
          />
        ) : (
          <SignToVoicePanel
            currentSign={currentSign}
            signConfidence={signConfidence}
            glosses={glosses}
            currentSentence={currentSentence}
            isSpeaking={isSpeaking}
            classifierStatus={classifierStatus}
            onClearGlosses={() => { glossBufferRef.current?.clear(); setGlosses([]) }}
            onClearSentence={() => setCurrentSentence('')}
          />
        )}
      </main>

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  )
}
