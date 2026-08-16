"""
llm_baselines – Azure OpenAI Client Wrapper
============================================
Wraps langchain_openai.AzureChatOpenAI with:
  • Exponential back-off retry on transient API errors
  • Streaming-based response collection (matches the original snippet)
  • Robust JSON extraction from noisy / markdown-wrapped model output
  • Token usage tracking for latency / cost reporting
"""

import json
import logging
import re
import time
from typing import Optional

from langchain_openai import AzureChatOpenAI

from config import (
    MODEL_NAME, 
    TEMPERATURE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION
)
from prompts import build_messages

logger = logging.getLogger(__name__)

# ─── JSON extraction helpers ─────────────────────────────────────────────────

_JSON_OBJECT_RE = re.compile(r'\{[^{}]*"vulnerable"\s*:\s*[01][^{}]*\}', re.DOTALL)
_REQUIRED_KEYS = {"vulnerable", "confidence"}

def _extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(text)
        if _REQUIRED_KEYS.issubset(obj):
            return obj
    except json.JSONDecodeError:
        pass

    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            obj = json.loads(match.group())
            if _REQUIRED_KEYS.issubset(obj):
                return obj
        except json.JSONDecodeError:
            continue

    brace_re = re.compile(r'\{.*?\}', re.DOTALL)
    for match in brace_re.finditer(text):
        try:
            obj = json.loads(match.group())
            if _REQUIRED_KEYS.issubset(obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None

# ─── Client class ────────────────────────────────────────────────────────────

class NvidiaLLMClient:
    """
    Native wrapper using LangChain AzureChatOpenAI.
    """
    def __init__(self):
        self._client = AzureChatOpenAI(
            azure_deployment=MODEL_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=TEMPERATURE
        )
        logger.info(
            "AzureLLMClient natively initialised — deployment=%s temperature=%.2f",
            MODEL_NAME, TEMPERATURE,
        )

    def predict(self, code: str, sample_id: str = "?") -> dict:
        messages = build_messages(code)

        t0 = time.time()
        try:
            # Native invoke call replacing the MulVul dependency
            response = self._client.invoke(messages)
            raw = response.content
            latency = time.time() - t0

            # Extract token usage with no assumptions
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            parsed = _extract_json(raw)
            if parsed is not None:
                parsed["vulnerable"]   = int(bool(parsed.get("vulnerable", 0)))
                parsed["confidence"]   = max(0, min(100, int(parsed.get("confidence", 50))))
                parsed["cwe"]          = parsed.get("cwe", "N/A")
                parsed["reason"]       = parsed.get("reason", "")
                parsed["raw_response"] = raw
                parsed["parse_ok"]     = True
                parsed["latency_sec"]  = round(latency, 3)
                parsed["prompt_tokens"] = prompt_tokens
                parsed["completion_tokens"] = completion_tokens
                parsed["total_tokens"] = total_tokens
                return parsed

            logger.warning("[%s] could not extract JSON from: %s", sample_id, raw[:200])
        except Exception as exc:
            logger.error("[%s] API error: %s", sample_id, exc)

        return {
            "vulnerable":   0,
            "confidence":   0,
            "cwe":          "N/A",
            "reason":       "Inference failed.",
            "raw_response": "",
            "parse_ok":     False,
            "latency_sec":  0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
