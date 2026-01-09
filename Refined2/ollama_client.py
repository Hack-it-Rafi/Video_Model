import requests
import json
from typing import List, Tuple

OLLAMA_BASE = 'http://localhost:11434'


def ollama_summarize(steps: List[Tuple[str, float]], model_name: str = 'llama3.2') -> str:
    """Call local Ollama server to produce a natural-language summary for `steps`.
    Tries /api/chat (native Ollama endpoint) with proper formatting.
    Returns the assistant text.
    Steps can be (label, conf) or (label, conf, meta).
    """
    system_prompt = "You are an assistant that summarizes UI action sequences into a short natural description and a numbered step list. Be concise."
    user_text = 'Given the following steps (label, optional metadata and confidence), write a 2-4 sentence summary and then a numbered list with each step:\n\n'
    for i, step in enumerate(steps):
        if len(step) == 3:
            lbl, conf, meta = step
            meta_info = f" [{meta}]" if meta else ""
            user_text += f"{i+1}. {lbl} (confidence {conf:.2f}){meta_info}\n"
        else:
            lbl, conf = step
            user_text += f"{i+1}. {lbl} (confidence {conf:.2f})\n"

    # Try native Ollama chat API first
    try:
        chat_payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "stream": False
        }
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=chat_payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        if 'message' in j and 'content' in j['message']:
            return j['message']['content']
        return str(j)
    except Exception as e:
        # Fallback to generate API
        try:
            payload = {
                "model": model_name,
                "prompt": system_prompt + "\n\nUser:\n" + user_text,
                "stream": False
            }
            r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            if 'response' in j:
                return j['response']
            return str(j)
        except Exception as e2:
            raise RuntimeError(f"Ollama request failed: {e2}")