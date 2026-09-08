"""Serverless handler for the Learnigence AI Product Coach front end.

Deploy alongside the static page (e.g. Vercel `api/` directory). The browser
posts to /api/coach; the Gemini key stays server-side in GEMINI_API_KEY.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_INSTRUCTION = (
    "You are an entrepreneurship coach teaching users to build AI driven products. "
    "Whatever they ask, your job is to help them discover the answer themself by "
    "encouraging them to go back to their beliefs about how they create value for "
    "customers, the idea they must break their work into testable components, and then "
    "build products that they iterate on through evaluation of customer interactions. "
    "Your goal is to help the user walk away empowered that they can clarify their own "
    "AI design choices.\n\nStyle:\nOnly give 1 to 2 line answers so the user can easily "
    "manage the cognitive load."
)

GENERATION_CONFIG = {
    "temperature": 1,
    "max_output_tokens": 65536,
    "top_p": 0.95,
    "thinking_level": "high",
}


def _flatten(contents):
    """Turn the front end's contents array into a single input string."""
    lines = []
    for turn in contents or []:
        role = "Coach" if turn.get("role") == "model" else "User"
        text = " ".join(p.get("text", "") for p in turn.get("parts", []))
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n".join(lines)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        interaction = client.interactions.create(
            model=body.get("model", "models/gemini-3.1-pro-preview"),
            input=_flatten(body.get("contents")),
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=GENERATION_CONFIG,
        )

        payload = json.dumps({"outputText": interaction.output_text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
