import React from 'react'

const STATUS = {
  online:   { dot: 'bg-emerald-500', label: 'Classifier online' },
  offline:  { dot: 'bg-red-500',     label: 'Classifier offline — start Python server on :8001' },
  checking: { dot: 'bg-yellow-400 animate-pulse', label: 'Checking classifier…' },
}

export default function SignToVoicePanel({
  currentSign,
  signConfidence,
  glosses,
  currentSentence,
  isSpeaking,
  classifierStatus,
  onClearGlosses,
  onClearSentence,
}) {
  const pct = Math.round((signConfidence || 0) * 100)
  const s = STATUS[classifierStatus] || STATUS.checking

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Classifier status bar */}
      <div className="flex items-center gap-2.5 bg-gray-800/70 border border-gray-700/50 rounded-xl px-4 py-2.5">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${s.dot}`} />
        <span className="text-xs text-gray-400 leading-tight">{s.label}</span>
      </div>

      {/* Current detected sign */}
      <div className="bg-gray-800 rounded-2xl p-5 text-center shadow-lg flex-shrink-0">
        <p className="text-xs uppercase tracking-widest text-gray-400 mb-2">Detected Word</p>
        <div className="min-h-[64px] flex items-center justify-center">
          {currentSign ? (
            <span className="text-5xl font-black text-emerald-400 uppercase tracking-wide">{currentSign}</span>
          ) : (
            <span className="text-gray-600 text-3xl">—</span>
          )}
        </div>
        <div className="mt-3 w-full bg-gray-700 rounded-full h-1.5">
          <div
            className="h-1.5 rounded-full bg-emerald-500 transition-all duration-200"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-gray-500 text-xs mt-1">{pct}% confidence</p>
      </div>

      {/* Gloss buffer — accumulating signs */}
      <div className="bg-gray-800 rounded-2xl p-4 shadow-lg flex-shrink-0">
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs uppercase tracking-widest text-gray-400">Signs Queued</p>
          <button
            onClick={onClearGlosses}
            className="text-xs text-gray-600 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5 min-h-[38px] items-start">
          {glosses.length === 0 ? (
            <p className="text-gray-600 text-sm italic">Start signing…</p>
          ) : (
            glosses.map((g, i) => (
              <span
                key={i}
                className="bg-emerald-950 text-emerald-300 border border-emerald-700/60 text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wide"
              >
                {g}
              </span>
            ))
          )}
        </div>
        <p className="text-gray-700 text-xs mt-2">Pause 1.5s after signing to speak</p>
      </div>

      {/* Sentence output */}
      <div className="bg-gray-800 rounded-2xl p-5 flex-1 flex flex-col shadow-lg min-h-0">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs uppercase tracking-widest text-gray-400">Spoken Sentence</p>
          {isSpeaking && (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Speaking…
            </span>
          )}
        </div>
        <div className="flex-1 bg-gray-900 rounded-xl p-4 overflow-y-auto">
          {currentSentence === '...' ? (
            <p className="text-gray-400 text-base animate-pulse">Building sentence…</p>
          ) : currentSentence ? (
            <p className="text-white text-xl font-semibold break-words leading-relaxed">{currentSentence}</p>
          ) : (
            <p className="text-gray-600 italic text-sm">Sentence will appear here after you pause signing…</p>
          )}
        </div>
        <button
          onClick={onClearSentence}
          className="mt-3 w-full bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm font-semibold py-2 rounded-xl transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  )
}
