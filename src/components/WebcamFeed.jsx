import React, { forwardRef } from 'react'
import Webcam from 'react-webcam'

const VIDEO_CONSTRAINTS = { facingMode: 'user', width: 640, height: 480 }

const WebcamFeed = forwardRef(function WebcamFeed(
  { isReady, canvasRef, mode, currentSign, signConfidence },
  ref
) {
  const pct = Math.round((signConfidence || 0) * 100)

  return (
    <div className="relative rounded-2xl overflow-hidden shadow-2xl bg-gray-900 aspect-video">
      <Webcam
        ref={ref}
        mirrored
        videoConstraints={VIDEO_CONSTRAINTS}
        className="w-full h-full object-cover block"
      />

      {/* Canvas overlay for hand landmark skeleton (reverse mode) */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Sign label HUD — shown in reverse mode when a sign is detected */}
      {mode === 'reverse' && currentSign && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/75 backdrop-blur-sm rounded-2xl px-5 py-2.5 text-center pointer-events-none z-10 border border-emerald-700/40">
          <p className="text-white text-2xl font-black uppercase tracking-wider leading-none">{currentSign}</p>
          <p className="text-emerald-400 text-xs mt-1 font-medium">{pct}% confidence</p>
        </div>
      )}

      {/* Mode badge */}
      <div className={`absolute bottom-3 left-3 text-xs font-semibold px-2.5 py-1 rounded-full border ${
        mode === 'forward'
          ? 'bg-indigo-900/70 border-indigo-600/50 text-indigo-300'
          : 'bg-emerald-900/70 border-emerald-600/50 text-emerald-300'
      }`}>
        {mode === 'forward' ? '✋ Sign → Text' : '🔊 Sign → Voice'}
      </div>

      {/* Loading overlay */}
      {!isReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/85 gap-3 z-20">
          <div className={`w-10 h-10 border-4 border-t-transparent rounded-full animate-spin ${
            mode === 'forward' ? 'border-indigo-500' : 'border-emerald-500'
          }`} />
          <p className="text-gray-300 text-sm font-medium">
            {mode === 'reverse' ? 'Loading hand detector…' : 'Loading gesture model…'}
          </p>
        </div>
      )}
    </div>
  )
})

export default WebcamFeed
