"""Final-pass transcript cleanup backed by a local Ollama model."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

try:
    from ollama import Client as OllamaPackageClient
except ImportError:  # pragma: no cover - exercised indirectly via runtime error path
    OllamaPackageClient = None


DEFAULT_SYSTEM_PROMPT = (
    "You clean speech-to-text transcripts. "
    "Preserve meaning, preserve the speaker's wording when possible, and never invent facts. "
    "Fix punctuation, capitalization, spacing, and obvious recognition mistakes only when the context is clear. "
    "Return only the cleaned transcript text."
)


class OllamaClient:
    """Thin wrapper around the Ollama Python client."""

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        timeout: float = 5.0,
        client: Optional[Any] = None,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self._client = client or self._build_client()

    def _build_client(self) -> Any:
        if OllamaPackageClient is None:
            raise RuntimeError(
                "The 'ollama' package is not installed. Install dependencies before enabling Ollama post-processing."
            )

        return OllamaPackageClient(host=self.endpoint, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> str:
        """Generate a deterministic cleanup response from Ollama."""
        response = self._client.generate(
            model=self.model_name,
            prompt=prompt,
            system=system,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return self._extract_text(response)

    def warm(self) -> None:
        """Issue a tiny request to encourage the model to load before first use."""
        self.generate("Ready.", max_tokens=1, temperature=0.0, system="Reply with OK.")

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, Mapping):
            return str(response.get("response", "")).strip()

        return str(getattr(response, "response", "")).strip()


class LLMPostProcessor:
    """Single-pass transcript cleanup using a local Ollama model."""

    def __init__(
        self,
        client: OllamaClient,
        user_vocab: Optional[Mapping[str, str]] = None,
    ):
        self.client = client
        self.user_vocab = dict(user_vocab or {})

    def process(self, text: str) -> str:
        """Return cleaned transcript text, or the original text on failure."""
        cleaned_input = text.strip()
        if not cleaned_input:
            return cleaned_input

        prompt = self.build_prompt(cleaned_input)
        start_time = time.time()
        try:
            result = self.client.generate(
                prompt,
                max_tokens=max(len(cleaned_input.split()) * 3, 64),
                temperature=0.0,
                system=DEFAULT_SYSTEM_PROMPT,
            )
        except Exception as exc:
            print(f"⚠️ Ollama post-processing failed: {exc}")
            return cleaned_input

        elapsed = time.time() - start_time
        if result:
            print(f"LLM post-processing completed in {elapsed:.2f}s")
            return result.strip()

        print("⚠️ Ollama post-processing returned empty output; using original transcript.")
        return cleaned_input

    def build_prompt(self, text: str) -> str:
        """Build the prompt for the final transcript cleanup pass."""
        sections = [
            "Clean this transcript while preserving meaning and wording:",
            text,
        ]

        if self.user_vocab:
            vocab_lines = ["Preferred vocabulary and corrections:"]
            for source, target in self.user_vocab.items():
                vocab_lines.append(f"- {source} -> {target}")
            sections.append("\n".join(vocab_lines))

        return "\n\n".join(sections)