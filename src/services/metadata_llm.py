import os
import asyncio
import logging
from typing import Optional, Dict

import httpx
import google.generativeai as genai
from ollama import Client

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """LLM-backed extractor for paper metadata (title, authors, year)."""

    def __init__(self):
        self.llm_type = os.getenv('LLM_TYPE', 'ollama').lower()
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY', '')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'deepseek-v3.1:671b')  # This is the correct model
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"

        # Remove all google.generativeai/genai logic and Gemini config/refs.

    async def extract(self, filename: str, text_sample: str) -> Optional[Dict]:
        """
        Use the configured LLM to extract title, authors, and year from a small text sample (ideally first 1-2 pages).
        Returns a dict {title, authors, year} or None on failure.
        """
        prompt = self._build_prompt(filename, text_sample)

        try:
            if self.llm_type == "ollama":
                return await self._call_ollama(prompt)
            if self.llm_type == "deepseek":
                return await self._call_deepseek(prompt)
        except Exception as e:
            logger.error(f"LLM metadata extraction failed: {e}")

        return None

    def _build_prompt(self, filename: str, text_sample: str) -> str:
        return (
            "You are extracting bibliographic metadata from the beginning of a research paper.\n"
            "Given the filename and the first page(s) of text, return a strict JSON object with keys: "
            "title (string), authors (string; comma-separated full names), year (integer or null).\n"
            "Rules:\n"
            "- Prefer the paper's actual title over headers like arXiv IDs or dates.\n"
            "- Authors should be human names; do not return single stop-words like 'The'.\n"
            "- Year should be the publication year if clear, else null.\n"
            "- Respond with ONLY the JSON, no extra text.\n\n"
            f"FILENAME: {filename}\n"
            "TEXT:\n"
            f"{text_sample[:4000]}\n"
        )

    async def _call_ollama(self, prompt: str) -> Optional[Dict]:
        """Use Ollama Cloud to get the metadata extraction."""
        import asyncio
        host = "https://ollama.com"
        api_key = self.ollama_api_key
        if not api_key:
            return None
        client = Client(
            host=host,
            headers={'Authorization': 'Bearer ' + api_key}
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        output = []
        loop = asyncio.get_event_loop()
        def get_response():
            for part in client.chat(self.ollama_model, messages=messages, stream=False):
                print('Ollama part:', part)  # For debugging
                content = ''
                # Try extracting content
                try:
                    content = part['message']['content']
                except (KeyError, TypeError):
                    try:
                        content = part['content']
                    except Exception:
                        # fallback, just capture something
                        content = str(part)
                if content:
                    output.append(content)
        await loop.run_in_executor(None, get_response)
        content = ''.join(output).strip()
        return self._parse_json(content)

    async def _call_deepseek(self, prompt: str) -> Optional[Dict]:
        if not self.deepseek_api_key:
            return None
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                self.deepseek_api_url,
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return self._parse_json(content)

    def _parse_json(self, raw: str) -> Optional[Dict]:
        import json
        # Attempt to extract JSON if extra text leaked
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Fallback: try to find first {...}
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception:
            return None


# Singleton dependency
_singleton = LLMMetadataExtractor()

async def get_llm_metadata_extractor() -> LLMMetadataExtractor:
    return _singleton


