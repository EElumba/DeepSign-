// Copy this file to config.js and fill in your values.
// config.js is gitignored — never commit it.

export const CONFIG = {
  // Get a free key at https://console.deepgram.com
  // Leave empty to use the browser's Web Speech API fallback instead.
  DEEPGRAM_API_KEY:       '',

  // Snowflake logging — disabled for now (post-MVP)
  // SNOWFLAKE_ACCOUNT:   '',       // e.g. 'xy12345.us-east-1'
  // SNOWFLAKE_JWT:       '',       // JWT for key-pair auth
  // SNOWFLAKE_DATABASE:  'ACCESSLINK',
  // SNOWFLAKE_SCHEMA:    'PUBLIC',
  // SNOWFLAKE_WAREHOUSE: 'COMPUTE_WH',
  // LOG_BATCH_SIZE:      20,
  // LOG_FLUSH_INTERVAL_MS: 10000,

  // Pipeline timing
  SIGN_DELAY_MS:          400,
  AUDIO_CHUNK_MS:         250,

  // Display
  SHOW_CAPTIONS:          true,
  CAMERA_FACING:          'user',
  AVATAR_POSITION:        'bottom-right',
  AVATAR_SCALE:           0.38,
  CAPTION_HIGHLIGHT_COLOR:'#1D9E75',

  // Debug
  DEBUG_MODE:             false,
};
