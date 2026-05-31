const PERSIST_KEYS = [
  'ANTHROPIC_API_KEY',
  'ELEVENLABS_API_KEY',
  'ELEVENLABS_VOICE_ID',
  'CONFIDENCE_THRESHOLD',
  'CLASSIFIER_ENDPOINT',
]

function loadStored() {
  try {
    return JSON.parse(localStorage.getItem('deepsign_config') || '{}')
  } catch {
    return {}
  }
}

const env = import.meta.env

export let CONFIG = {
  // Forward mode
  HOLD_MS:                1200,
  MIN_CONFIDENCE:         0.72,

  // Reverse mode — Claude API
  ANTHROPIC_API_KEY:      env.VITE_ANTHROPIC_API_KEY      || '',
  CLAUDE_MODEL:           'claude-sonnet-4-20250514',

  // Reverse mode — ElevenLabs TTS
  ELEVENLABS_API_KEY:     env.VITE_ELEVENLABS_API_KEY     || '',
  ELEVENLABS_VOICE_ID:    env.VITE_ELEVENLABS_VOICE_ID    || 'EXAVITQu4vr4xnSDxMaL',
  ELEVENLABS_MODEL_ID:    'eleven_flash_v2_5',

  // Reverse mode — TGCN classifier
  WLASL_VOCAB_SIZE:       100,
  CONFIDENCE_THRESHOLD:   0.75,
  N_FRAMES:               30,
  SIGNING_PAUSE_MS:       1500,
  CLASSIFIER_ENDPOINT:    env.VITE_CLASSIFIER_ENDPOINT    || 'http://localhost:8001/classify',

  TTS_AUTOPLAY:           true,
  TTS_VOLUME:             1.0,
  DEBUG_MODE:             false,
}

Object.assign(CONFIG, loadStored())

export function updateConfig(updates) {
  Object.assign(CONFIG, updates)
  const toSave = {}
  PERSIST_KEYS.forEach(k => { if (k in CONFIG) toSave[k] = CONFIG[k] })
  localStorage.setItem('deepsign_config', JSON.stringify(toSave))
}
