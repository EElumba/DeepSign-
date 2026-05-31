import React, { useState } from 'react'
import { CONFIG, updateConfig } from '../config'

export default function SettingsModal({ onClose }) {
  const [form, setForm] = useState({
    ANTHROPIC_API_KEY:    CONFIG.ANTHROPIC_API_KEY,
    ELEVENLABS_API_KEY:   CONFIG.ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID:  CONFIG.ELEVENLABS_VOICE_ID,
    CLASSIFIER_ENDPOINT:  CONFIG.CLASSIFIER_ENDPOINT,
    CONFIDENCE_THRESHOLD: CONFIG.CONFIDENCE_THRESHOLD,
  })

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const save = () => {
    updateConfig({
      ...form,
      CONFIDENCE_THRESHOLD: parseFloat(form.CONFIDENCE_THRESHOLD) || 0.75,
    })
    onClose()
  }

  return (
    <div
      className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-md p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white text-lg leading-none transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <Field label="Anthropic API Key" hint="Enables ASL gloss → natural English (optional)">
            <input
              type="password"
              value={form.ANTHROPIC_API_KEY}
              onChange={e => set('ANTHROPIC_API_KEY', e.target.value)}
              placeholder="sk-ant-…"
              className="w-full bg-gray-800 border border-gray-600 rounded-xl px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </Field>

          <Field label="ElevenLabs API Key" hint="Leave blank to use Web Speech API (browser TTS)">
            <input
              type="password"
              value={form.ELEVENLABS_API_KEY}
              onChange={e => set('ELEVENLABS_API_KEY', e.target.value)}
              placeholder="xi_…"
              className="w-full bg-gray-800 border border-gray-600 rounded-xl px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </Field>

          <Field label="ElevenLabs Voice ID">
            <input
              type="text"
              value={form.ELEVENLABS_VOICE_ID}
              onChange={e => set('ELEVENLABS_VOICE_ID', e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </Field>

          <Field label="Classifier Endpoint" hint="Python TGCN server (python/sign_classifier.py)">
            <input
              type="text"
              value={form.CLASSIFIER_ENDPOINT}
              onChange={e => set('CLASSIFIER_ENDPOINT', e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors font-mono"
            />
          </Field>

          <Field label="Confidence Threshold" hint="Minimum classifier confidence to accept (0.0–1.0)">
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={form.CONFIDENCE_THRESHOLD}
                onChange={e => set('CONFIDENCE_THRESHOLD', e.target.value)}
                className="flex-1 accent-emerald-500"
              />
              <span className="text-sm text-white font-mono w-10 text-right">
                {Number(form.CONFIDENCE_THRESHOLD).toFixed(2)}
              </span>
            </div>
          </Field>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 py-2.5 bg-gray-800 hover:bg-gray-700 rounded-xl text-sm font-semibold transition-colors text-gray-300"
          >
            Cancel
          </button>
          <button
            onClick={save}
            className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-sm font-semibold transition-colors text-white"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}

function Field({ label, hint, children }) {
  return (
    <div>
      <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider block mb-1.5">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-gray-600 mt-1">{hint}</p>}
    </div>
  )
}
