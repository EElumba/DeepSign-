import { CONFIG } from '../config.js';

export class AudioCapture {
  constructor({ onFinal, onInterim }) {
    this._onFinal = onFinal;
    this._onInterim = onInterim;
    this._socket = null;
    this._recorder = null;
    this._recognition = null;
  }

  async start() {
    const useDeepgram = CONFIG.DEEPGRAM_API_KEY && CONFIG.DEEPGRAM_API_KEY.trim().length > 0;
    if (useDeepgram) {
      await this._startDeepgram();
    } else {
      this._startWebSpeech();
    }
  }

  async _startDeepgram() {
    const url =
      'wss://api.deepgram.com/v1/listen' +
      '?model=nova-2&language=en-US&interim_results=true&punctuate=false';

    this._socket = new WebSocket(url, ['token', CONFIG.DEEPGRAM_API_KEY]);

    await new Promise((resolve, reject) => {
      this._socket.onopen = resolve;
      this._socket.onerror = () => reject(new Error('Deepgram WebSocket failed'));
      setTimeout(() => reject(new Error('Deepgram connection timeout')), 5000);
    });

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    this._recorder.ondataavailable = e => {
      if (this._socket?.readyState === WebSocket.OPEN) this._socket.send(e.data);
    };
    this._recorder.start(CONFIG.AUDIO_CHUNK_MS);

    this._socket.onmessage = msg => {
      try {
        const data = JSON.parse(msg.data);
        const transcript = data?.channel?.alternatives?.[0]?.transcript;
        if (!transcript) return;
        if (CONFIG.DEBUG_MODE) console.log('[Deepgram]', { transcript, isFinal: data.is_final });
        if (data.is_final) this._onFinal(transcript);
        else this._onInterim(transcript);
      } catch { /* ignore malformed frames */ }
    };

    this._socket.onclose = () => {
      console.warn('[AudioCapture] Deepgram closed — falling back to Web Speech');
      this._startWebSpeech();
    };

    this._socket.onerror = () => {
      console.warn('[AudioCapture] Deepgram error — falling back to Web Speech');
      this._startWebSpeech();
    };
  }

  _startWebSpeech() {
    const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SR) {
      console.error('[AudioCapture] No speech recognition available in this browser');
      return;
    }

    this._recognition = new SR();
    this._recognition.continuous = true;
    this._recognition.interimResults = true;
    this._recognition.lang = 'en-US';

    this._recognition.onresult = e => {
      for (const result of e.results) {
        const text = result[0].transcript;
        if (CONFIG.DEBUG_MODE) console.log('[WebSpeech]', { text, isFinal: result.isFinal });
        if (result.isFinal) this._onFinal(text);
        else this._onInterim(text);
      }
    };

    this._recognition.onerror = e => {
      if (e.error !== 'no-speech') console.warn('[AudioCapture] Web Speech error:', e.error);
    };

    // Auto-restart on end to keep session alive
    this._recognition.onend = () => {
      if (this._recognition) this._recognition.start();
    };

    this._recognition.start();
    console.info('[AudioCapture] Using Web Speech API fallback');
  }

  stop() {
    if (this._recorder) {
      this._recorder.stop();
      this._recorder.stream?.getTracks().forEach(t => t.stop());
      this._recorder = null;
    }
    if (this._socket) {
      this._socket.onclose = null;
      this._socket.close();
      this._socket = null;
    }
    if (this._recognition) {
      this._recognition.onend = null;
      this._recognition.stop();
      this._recognition = null;
    }
  }
}
