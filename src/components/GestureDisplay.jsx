import React from 'react'

export default function GestureDisplay({ gesture, confidence, sentence, onBackspace, onClear, onSpace }) {
  const pct = Math.round(confidence * 100)

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Live sign card */}
      <div className="bg-gray-800 rounded-2xl p-6 text-center shadow-lg flex-shrink-0">
        <p className="text-xs uppercase tracking-widest text-gray-400 mb-2">Detected Sign</p>
        <div className="text-8xl font-black text-white leading-none min-h-[96px] flex items-center justify-center">
          {gesture || <span className="text-gray-600 text-5xl">—</span>}
        </div>
        <div className="mt-4 w-full bg-gray-700 rounded-full h-2">
          <div
            className="h-2 rounded-full bg-indigo-500 transition-all duration-150"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-gray-500 text-xs mt-1">{pct}% confidence</p>
      </div>

      {/* Sentence builder */}
      <div className="bg-gray-800 rounded-2xl p-6 flex-1 flex flex-col shadow-lg min-h-0">
        <p className="text-xs uppercase tracking-widest text-gray-400 mb-3">Sentence</p>
        <div className="flex-1 bg-gray-900 rounded-xl p-4 overflow-y-auto">
          {sentence ? (
            <p className="text-white text-2xl font-semibold break-words leading-relaxed">{sentence}</p>
          ) : (
            <p className="text-gray-600 italic text-lg">Hold a sign to start…</p>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 mt-4">
          <button
            onClick={onSpace}
            className="col-span-1 bg-gray-700 hover:bg-gray-600 active:bg-gray-500 text-white text-sm font-semibold py-2 px-3 rounded-xl transition-colors"
          >
            Space
          </button>
          <button
            onClick={onBackspace}
            className="col-span-1 bg-gray-700 hover:bg-gray-600 active:bg-gray-500 text-white text-sm font-semibold py-2 px-3 rounded-xl transition-colors"
          >
            ⌫ Back
          </button>
          <button
            onClick={onClear}
            className="col-span-1 bg-red-700 hover:bg-red-600 active:bg-red-500 text-white text-sm font-semibold py-2 px-3 rounded-xl transition-colors"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  )
}
