// Serverless handler for the Learnigence AI Product Coach front end.
// Place at api/coach.js in a Vercel project. Set GEMINI_API_KEY in
// Project Settings -> Environment Variables, then redeploy.
// No dependencies and no requirements.txt: calls the Gemini REST API directly.

const SYSTEM_INSTRUCTION = `You are an entrepreneurship coach teaching users to build AI driven products. Whatever they ask, your job is to help them discover the answer themself by encouraging them to go back to their beliefs about how they create value for customers, the idea they must break their work into testable components, and then build products that they iterate on through evaluation of customer interactions. Your goal is to help the user walk away empowered that they can clarify their own AI design choices.

Style:
Only give 1 to 2 line answers so the user can easily manage the cognitive load.`;

const MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"];

async function callGemini(key, model, contents) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
      contents,
      generationConfig: { temperature: 1, topP: 0.95, maxOutputTokens: 8192 },
    }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error((data.error && data.error.message) || `Gemini returned ${res.status}`);
  }

  const parts =
    data.candidates &&
    data.candidates[0] &&
    data.candidates[0].content &&
    data.candidates[0].content.parts;
  return (parts || []).map((p) => p.text).filter(Boolean).join("\n").trim();
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Use POST." });
  }

  const key = process.env.GEMINI_API_KEY;
  if (!key) {
    return res.status(500).json({ error: "GEMINI_API_KEY is not set on this deployment." });
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const contents = body.contents || [];
    const requested = String(body.model || "").replace("models/", "");
    const candidates = requested ? [requested, ...MODELS] : MODELS;

    let lastError;
    for (const model of candidates) {
      try {
        const text = await callGemini(key, model, contents);
        if (text) return res.status(200).json({ outputText: text, model });
        lastError = new Error("Empty response.");
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError;
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
