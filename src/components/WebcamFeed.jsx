import React, { forwardRef } from 'react'
import Webcam from 'react-webcam'

const VIDEO_CONSTRAINTS = { facingMode: 'user', width: 640, height: 480 }

const WebcamFeed = forwardRef(function WebcamFeed({ isReady }, ref) {
  return (
    <div className="relative rounded-2xl overflow-hidden shadow-2xl bg-gray-900">
      <Webcam
        ref={ref}
        mirrored
        videoConstraints={VIDEO_CONSTRAINTS}
        className="w-full block"
      />
      {!isReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/80 gap-3">
          <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-300 text-sm font-medium">Loading gesture model…</p>
        </div>
      )}
    </div>
  )
})

export default WebcamFeed
