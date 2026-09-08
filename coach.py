"""Serverless handler for the Learnigence AI Product Coach front end.

Place at api/coach.py in a Vercel project. Requires a root requirements.txt
containing `google-genai`, and a GEMINI_API_KEY environment variable.
The browser posts to /api/coach; the key never leaves the server.
"""

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler

from google import genai
from google.genai import types

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

FALLBACK_MODEL = "gemini-2.5-pro"


def _flatten(contents):
    """Turn the front end's contents array into a single input string."""
    lines = []
    for turn in contents or []:
        role = "Coach" if turn.get("role") == "model" else "User"
        text = " ".join(p.get("text", "") for p in turn.get("parts", []))
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n".join(lines)


def _answer(client, model, prompt):
    """Prefer the Interactions API; fall back to generate_content if the
    installed SDK version does not expose it."""
    if hasattr(client, "interactions"):
        try:
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={
                    "temperature": 1,
                    "max_output_tokens": 65536,
                    "top_p": 0.95,
                    "thinking_level": "high",
                },
            )
            return interaction.output_text
        except Exception:
            pass

    response = client.models.generate_content(
        model=model.replace("models/", ""),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )
    return response.text


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            self._send(500, {"error": "GEMINI_API_KEY is not set on this deployment."})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            prompt = _flatten(request.get("contents"))
            model = request.get("model", "models/gemini-3.1-pro-preview")

            client = genai.Client(api_key=key)
            try:
                text = _answer(client, model, prompt)
            except Exception:
                # Requested model unavailable on this key — retry on a stable one.
                text = _answer(client, FALLBACK_MODEL, prompt)

            self._send(200, {"outputText": (text or "").strip()})
        except Exception as exc:
            print(traceback.format_exc())
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
