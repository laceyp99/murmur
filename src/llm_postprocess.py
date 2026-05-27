"""Final-pass transcript cleanup backed by a local Ollama model."""

from __future__ import annotations

import re
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

DISALLOWED_PREFIXES = (
    "here's the cleaned transcript",
    "here is the cleaned transcript",
    "cleaned transcript:",
    "corrected transcript:",
    "certainly",
    "sure",
    "of course",
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

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Send a deterministic chat request to Ollama and extract assistant text."""
        response = self._client.chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return self._extract_chat_text(response)

    def is_model_available(self) -> bool:
        """Return whether the configured model is already available locally."""
        models_response = self._client.list()
        for model_name in self._extract_model_names(models_response):
            if model_name == self.model_name:
                return True
        return False

    def warm(self, keep_alive: str = "10m") -> bool:
        """Load an already-installed model into memory without downloading it."""
        if not self.is_model_available():
            print(
                f"⚠️ Ollama model '{self.model_name}' is not installed locally; skipping warmup."
            )
            return False

        self._client.chat(
            model=self.model_name,
            messages=[],
            stream=False,
            keep_alive=keep_alive,
        )
        return True

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, Mapping):
            return str(response.get("response", "")).strip()

        return str(getattr(response, "response", "")).strip()

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        if isinstance(response, Mapping):
            message = response.get("message", {})
            if isinstance(message, Mapping):
                return str(message.get("content", "")).strip()
            return ""

        message = getattr(response, "message", None)
        if message is None:
            return ""

        if isinstance(message, Mapping):
            return str(message.get("content", "")).strip()

        return str(getattr(message, "content", "")).strip()

    @staticmethod
    def _extract_model_names(response: Any) -> list[str]:
        if isinstance(response, Mapping):
            models = response.get("models", [])
        else:
            models = getattr(response, "models", [])

        model_names = []
        for model in models:
            if isinstance(model, Mapping):
                name = model.get("name") or model.get("model")
            else:
                name = getattr(model, "name", None) or getattr(model, "model", None)

            if name:
                model_names.append(str(name))

        return model_names


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

        messages = self.build_messages(cleaned_input)
        start_time = time.time()
        try:
            result = self.client.chat(
                messages,
                max_tokens=min(max(len(cleaned_input.split()) * 3, 64), 1024),
                temperature=0.0,
            )
        except Exception as exc:
            print(f"⚠️ Ollama post-processing failed: {exc}")
            return cleaned_input

        elapsed = time.time() - start_time
        normalized_result = self._normalize_output(result)
        if self._is_acceptable_output(normalized_result, cleaned_input):
            print(f"LLM post-processing completed in {elapsed:.2f}s")
            return normalized_result

        print(
            "⚠️ Ollama post-processing returned invalid output; using original transcript."
        )
        return cleaned_input

    def _normalize_output(self, text: str) -> str:
        """Strip common wrapper formatting from model output before validation."""
        normalized_text = text.strip()

        if normalized_text.startswith("```"):
            normalized_text = re.sub(r"^```[^\n]*\n?", "", normalized_text)
            normalized_text = re.sub(r"\n?```$", "", normalized_text)

        normalized_text = re.sub(r"\n\s*\n+", "\n\n", normalized_text).strip()

        quote_pairs = (('"', '"'), ("'", "'"), ("“", "”"))
        for opening_quote, closing_quote in quote_pairs:
            if (
                normalized_text.startswith(opening_quote)
                and normalized_text.endswith(closing_quote)
                and len(normalized_text) >= 2
            ):
                normalized_text = normalized_text[1:-1].strip()
                break

        return normalized_text

    def _is_acceptable_output(self, output_text: str, input_text: str) -> bool:
        """Accept only transcript-like model output and reject assistant/meta responses."""
        if not output_text:
            return False

        lowered_output = output_text.casefold()
        if lowered_output.startswith(DISALLOWED_PREFIXES):
            return False

        if len(output_text) > int(len(input_text) * 1.75) + 40:
            return False

        if re.search(r"(?m)^#{1,6}\s", output_text):
            return False

        if len(re.findall(r"(?m)^\s*(?:[-*]|\d+\.)\s+", output_text)) >= 1:
            return False

        if re.search(r"(?im)^\s*(user|assistant|system|transcript):", output_text):
            return False

        return True

    def build_messages(self, text: str) -> list[dict[str, str]]:
        """Build few-shot chat history for the final transcript cleanup pass."""
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_user_prompt(
                    "quick recap we met with jane from blue ridge data about the pilot the transcript may say brew ridge or blue rich but it should be blue ridge data jane asked if noah can send the intake link and the loom walkthrough by friday"
                ),
            },
            {
                "role": "assistant",
                "content": "Quick recap: We met with Jane from Blue Ridge Data about the pilot. Jane asked if Noah can send the intake link and the Loom walkthrough by Friday.",
            },
            {
                "role": "user",
                "content": self._build_user_prompt(text),
            },
        ]

        return messages

    def _build_user_prompt(self, text: str) -> str:
        """Build the user turn content for transcript cleanup."""
        sections = [
            "Clean this transcript while preserving meaning and wording.",
            "Return only the cleaned transcript text with no preamble or commentary.",
        ]

        if self.user_vocab:
            vocab_lines = ["Preferred vocabulary and corrections:"]
            for source, target in self.user_vocab.items():
                vocab_lines.append(f"- {source} -> {target}")
            sections.append("\n".join(vocab_lines))

        sections.append(f"Transcript to clean:\n{text.strip()}")
        return "\n\n".join(sections)
