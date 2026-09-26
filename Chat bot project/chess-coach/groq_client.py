"""Thin wrapper over Groq's OpenAI-compatible chat API."""

import requests

BASE_URL = "https://api.groq.com/openai/v1"

# Fallback list used when we can't reach the live /models endpoint.
# Groq rotates its catalogue, so the app prefers the live list.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SKIP_KEYWORDS = ("whisper", "tts", "guard", "embed", "vision-preview")


class GroqError(Exception):
    pass


def list_models(api_key: str):
    """Ask Groq which models this key can actually use."""
    if not api_key:
        raise GroqError("No API key set.")
    try:
        response = requests.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise GroqError(f"Network error contacting Groq: {exc}") from exc

    if response.status_code == 401:
        raise GroqError("Groq rejected that API key (401). Check it and try again.")
    if response.status_code != 200:
        raise GroqError(f"Groq returned {response.status_code}: {response.text[:200]}")

    ids = [m.get("id", "") for m in response.json().get("data", [])]
    chat_models = [
        m for m in ids
        if m and not any(word in m.lower() for word in SKIP_KEYWORDS)
    ]
    return sorted(chat_models) or FALLBACK_MODELS


def chat(api_key: str, model: str, messages, temperature: float = 0.3, max_tokens: int = 700) -> str:
    if not api_key:
        raise GroqError("No API key set.")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise GroqError(f"Network error contacting Groq: {exc}") from exc

    if response.status_code == 401:
        raise GroqError("Groq rejected that API key (401).")
    if response.status_code == 429:
        raise GroqError("Rate limited by Groq (429). Wait a moment and retry.")
    if response.status_code == 404:
        raise GroqError(f"Model '{model}' is not available on your account (404). Pick another one in Settings.")
    if response.status_code != 200:
        raise GroqError(f"Groq returned {response.status_code}: {response.text[:300]}")

    data = response.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError) as exc:
        raise GroqError(f"Unexpected response shape from Groq: {data}") from exc