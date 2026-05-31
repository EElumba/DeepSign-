import { CONFIG } from '../../config'

// Sends ASL gloss sequence to Claude API → natural English sentence.
// Falls back to raw gloss words when no API key is configured.
export class SentenceBuilder {
  async build(glosses) {
    if (!CONFIG.ANTHROPIC_API_KEY) {
      return glosses.join(' ').toLowerCase()
    }

    const prompt = `You are an ASL-to-English interpreter. Convert this sequence of ASL gloss words into a single natural, grammatically correct English sentence. Respond with ONLY the sentence — no explanation, no punctuation before the sentence, no quotes.

ASL glosses: ${glosses.join(' ')}

Natural English:`

    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': CONFIG.ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: CONFIG.CLAUDE_MODEL,
          max_tokens: 150,
          messages: [{ role: 'user', content: prompt }],
        }),
      })
      const data = await res.json()
      return data.content?.[0]?.text?.trim() || glosses.join(' ')
    } catch {
      return glosses.join(' ')
    }
  }
}
