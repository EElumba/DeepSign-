class Pcm16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputSampleRate = sampleRate;
    this.outputSampleRate = 16000;
    this.pending = [];
    this.phase = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const ratio = this.inputSampleRate / this.outputSampleRate;
    const outLength = Math.floor((channel.length - this.phase) / ratio);
    if (outLength <= 0) return true;

    const pcm = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const index = Math.floor(this.phase + i * ratio);
      const sample = Math.max(-1, Math.min(1, channel[index] || 0));
      pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }

    this.phase = (this.phase + outLength * ratio) - channel.length;
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor('pcm16-capture', Pcm16CaptureProcessor);
