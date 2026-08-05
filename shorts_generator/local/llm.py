"""Local LLM backend — OpenAI, Gemini, or DeepSeek, selected by LLM_PROVIDER."""
from ..config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_deepseek_key,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by the render service."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for the render service. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by the render service when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_deepseek_llm(prompt: str) -> str:
    """DeepSeek Chat Completions backend (OpenAI-compatible API) used by the render service.

    DeepSeek exposes an OpenAI-compatible endpoint, so we reuse the `openai`
    SDK but point it at the DeepSeek base URL.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for the render service with LLM_PROVIDER=deepseek. "
            "Install it with:\n    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(
        api_key=require_deepseek_key(),
        base_url=DEEPSEEK_BASE_URL,
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider == "deepseek":
        return call_deepseek_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'deepseek'."
    )
