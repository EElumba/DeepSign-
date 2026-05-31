import { CONFIG } from '../../config'

// ElevenLabs Flash v2.5 TTS with Web Speech API fallback.
export class TTSPlayer {
  constructor() {
    this._playing = false
  }

  async speak(text) {
    if (!text || this._playing) return
    this._playing = true
    try {
      if (CONFIG.ELEVENLABS_API_KEY) {
        await this._elevenLabs(text)
      } else {
        await this._webSpeech(text)
      }
    } finally {
      this._playing = false
    }
  }

  async _elevenLabs(text) {
    const res = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_VOICE_ID}/stream`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          model_id: CONFIG.ELEVENLABS_MODEL_ID,
          voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        }),
      }
    )
    const arrayBuffer = await res.arrayBuffer()
    const audioCtx = new AudioContext()
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
    const source = audioCtx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioCtx.destination)
    source.start()
    await new Promise(resolve => { source.onended = resolve })
  }

  _webSpeech(text) {
    return new Promise(resolve => {
      const utt = new SpeechSynthesisUtterance(text)
      utt.lang = 'en-US'
      utt.rate = 1.0
      utt.onend = resolve
      utt.onerror = resolve
      speechSynthesis.speak(utt)
    })
  }
}
