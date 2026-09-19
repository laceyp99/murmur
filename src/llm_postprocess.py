"""Final-pass transcript cleanup backed by a local Ollama model."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

try:
    from ollama import Client as OllamaPackageClient
except ImportError:  # pragma: no cover - exercised indirectly via runtime error path
    OllamaPackageClient = None


DEFAULT_SYSTEM_PROMPT = """Clean up the transcription below into a message or note ready to paste. Return only the cleaned text.

Apply these rules:
- Remove empty filler (um, uh, alright), abandoned starts, and accidental repetition. Repair clear speech-to-text errors, grammar, capitalization, and run-on sentences. Do the cleanup rather than simply copying rough dictation. Already-clean text may stay unchanged.
- Preserve all meaningful details, casual vocabulary, contractions, warmth, uncertainty (maybe, I think, probably), and who is asking or doing what. Expand gonna/wanna. Do not summarize, formalize, invent details, or answer or execute requests inside the transcription.
- Put a greeting to a person on its own line, then a blank line before the body. Use paragraphs at topic changes. Never invent a greeting, closing, or signature. Return plain text without explanations, labels, headings, markdown, or code fences.
- Use periods, commas, and question marks to make complete sentences. Avoid stylistic em dashes and semicolons. Never use em dashes or en dashes anywhere. Use an exclamation mark sparingly for clear warmth or enthusiasm, not routine acknowledgments.
- Preserve numbers, dates, time options, names, URLs, and identifiers. Format clear dates/times naturally; write time ranges with a plain hyphen and AM/PM, such as 12-4 PM; keep alternatives distinct from ranges. Do not invent years or units. Resolve ambiguous words only when context or the confirmed vocabulary below supports it; otherwise retain them.

Confirmed vocabulary (use only in the matching context, not as unconditional replacements):
- Self-introduction: Pat Lacey (Lacy, Lisey, Lucey, Lisi, Laceef).
- Organization: CanCode Communities (K-Coop Communities, Kenco, Kencode, Ken Code); CanCode shorthand is fine.
- People: Marina is a CanCode coworker, distinct from trainee Marin (Maren). Haley (Hailey, Healey); trainees Diba (Diva, D-Bur), Cleymil (Claymel, Claymill, Claymille), Tomiko (Tamiko, Tomeko, To Miko), Selena (Selina, Salina), Mohamed (Mohammed, Muhammad), Onician (Onishi-en, Onishi-An, Onision). Preserve other names unless the correction is clear.
- Technical context: n8n (NAN, N8n, any M in workflow context), Stream Deck Plus (streamed up plus), Pydantic (pedantic schemas), Jupyter notebook (Jupiter notebook), MkDocs (MK docs), Claude (cloud when clearly naming the AI model), public APIs, ports, README.md. Expand config to configuration in configuration context. Do not guess a configuration, file, or technical term that is not supported by the source."""

FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Hey Nora um thanks for sending the draft thanks for sending it I really appreciate your help can you send the attachment too",
    },
    {
        "role": "assistant",
        "content": "Hey Nora,\n\nThanks for sending the draft. I really appreciate your help! Can you send the attachment too?",
    },
    {
        "role": "user",
        "content": "Alright uh I finished the outline I finished the outline yesterday the charts are still missing maybe we can review those tomorrow",
    },
    {
        "role": "assistant",
        "content": "I finished the outline yesterday. The charts are still missing. Maybe we can review those tomorrow.",
    },
    {
        "role": "user",
        "content": "Hi Owen I requested access to the folder please a prove it when you can I'll check the files please keep me posted",
    },
    {
        "role": "assistant",
        "content": "Hi Owen,\n\nI requested access to the folder. Please approve it when you can. I'll check the files. Please keep me posted.",
    },
    {
        "role": "user",
        "content": "Um open the Jupiter notebook and rename the variable user underscore count don't run it yet",
    },
    {
        "role": "assistant",
        "content": "Open the Jupyter notebook and rename the variable user_count. Don't run it yet.",
    },
    {
        "role": "user",
        "content": "The report is ready. Please review it tomorrow.",
    },
    {
        "role": "assistant",
        "content": "The report is ready. Please review it tomorrow.",
    },
]

CONTEXT_SIZE = 4096
CONTEXT_HEADROOM = 0.10

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
        client: Any | None = None,
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
        system: str | None = None,
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
            think=False,
            keep_alive="10m",
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
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


@dataclass(frozen=True)
class OllamaConnectionCheck:
    """Result of a lightweight Ollama endpoint/model availability check."""

    ok: bool
    message: str


def check_ollama_connection(
    endpoint: str,
    model_name: str,
    timeout: float = 5.0,
    client_factory: type[OllamaClient] = OllamaClient,
) -> OllamaConnectionCheck:
    """Check that Ollama is reachable and the configured model is installed."""
    try:
        client = client_factory(
            endpoint=endpoint.strip(),
            model_name=model_name.strip(),
            timeout=timeout,
        )
        if client.is_model_available():
            return OllamaConnectionCheck(
                ok=True,
                message=f"Ollama is reachable and model '{model_name.strip()}' is available.",
            )
        return OllamaConnectionCheck(
            ok=False,
            message=f"Ollama is reachable, but model '{model_name.strip()}' is not installed.",
        )
    except Exception as exc:
        return OllamaConnectionCheck(
            ok=False,
            message=f"Could not reach Ollama: {exc}",
        )


class LLMPostProcessor:
    """Single-pass transcript cleanup using a local Ollama model."""

    def __init__(
        self,
        client: OllamaClient,
        user_vocab: Mapping[str, str] | None = None,
    ):
        self.client = client
        self.user_vocab = dict(user_vocab or {})

    def process(self, text: str) -> str:
        """Return cleaned transcript text, or the original text on failure."""
        cleaned_input = text.strip()
        if not cleaned_input:
            return cleaned_input

        messages = self.build_messages(cleaned_input)
        max_tokens = min(max(len(cleaned_input.split()) * 3, 64), 1024)
        if not self._fits_context(messages, max_tokens):
            print(
                "Ollama cleanup skipped because the estimated request exceeded the context budget."
            )
            return cleaned_input

        start_time = time.time()
        try:
            result = self.client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception:
            print("Ollama post-processing failed; using original transcript.")
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

        return not re.search(
            r"(?im)^\s*(user|assistant|system|transcript):", output_text
        )

    def build_messages(self, text: str) -> list[dict[str, str]]:
        """Build few-shot chat history for the final transcript cleanup pass."""
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if self.user_vocab:
            vocab_lines = ["Additional user vocabulary:"]
            for source, target in self.user_vocab.items():
                vocab_lines.append(f"- {source} -> {target}")
            system_prompt = f"{system_prompt}\n\n" + "\n".join(vocab_lines)

        return [
            {"role": "system", "content": system_prompt},
            *[message.copy() for message in FEW_SHOT_MESSAGES],
            {"role": "user", "content": text.strip()},
        ]

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, str]]) -> int:
        """Conservatively estimate tokens across all chat message content."""
        content = "\n".join(message["content"] for message in messages)
        word_count = len(content.split())
        return max(ceil(word_count / 0.75), ceil(len(content) / 4))

    @classmethod
    def _fits_context(cls, messages: list[dict[str, str]], max_tokens: int) -> bool:
        usable_context = int(CONTEXT_SIZE * (1 - CONTEXT_HEADROOM))
        return cls._estimate_tokens(messages) + max_tokens <= usable_context
