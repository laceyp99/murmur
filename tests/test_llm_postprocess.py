import hashlib
import json
from types import SimpleNamespace

from src.config import DEFAULT_OLLAMA_MODEL_NAME, DEFAULT_OLLAMA_TIMEOUT_SECONDS
from src.llm_postprocess import (
    DEFAULT_SYSTEM_PROMPT,
    FEW_SHOT_MESSAGES,
    LLMPostProcessor,
    OllamaClient,
    check_ollama_connection,
)

MODEL_NAME = DEFAULT_OLLAMA_MODEL_NAME
TIMEOUT_SECONDS = float(DEFAULT_OLLAMA_TIMEOUT_SECONDS)


def test_baseline_messages_match_evaluated_prompt_artifact_exactly():
    processor = LLMPostProcessor(client=object())
    messages = processor.build_messages("{{input}}")
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "8e5166403532f02f7a262339ed1db4f1993fc0ee34c16e329843f6966599e7c2"
    )


class FakeOllamaPackageClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.list_response = {"models": []}
        self.chat_response = {"message": {"content": ""}}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def list(self):
        return self.list_response

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.chat_response


def test_ollama_client_generate_uses_model_and_options():
    fake_client = FakeOllamaPackageClient({"response": "Cleaned output."})
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name=MODEL_NAME,
        timeout=TIMEOUT_SECONDS,
        client=fake_client,
    )

    result = client.generate(
        "input text", max_tokens=42, temperature=0.0, system="system"
    )

    assert result == "Cleaned output."
    assert fake_client.calls == [
        {
            "model": MODEL_NAME,
            "prompt": "input text",
            "system": "system",
            "options": {
                "temperature": 0.0,
                "num_predict": 42,
            },
        }
    ]


def test_ollama_client_generate_extracts_object_response():
    fake_client = FakeOllamaPackageClient(SimpleNamespace(response="Object response."))
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name=MODEL_NAME,
        client=fake_client,
    )

    assert client.generate("input text") == "Object response."


def test_ollama_client_chat_uses_messages_and_options():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {"message": {"content": "Cleaned output."}}
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name=MODEL_NAME,
        timeout=TIMEOUT_SECONDS,
        client=fake_client,
    )

    result = client.chat(
        [{"role": "user", "content": "input text"}],
        max_tokens=42,
        temperature=0.0,
    )

    assert result == "Cleaned output."
    assert fake_client.calls == [
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "input text"}],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 42,
                "num_ctx": 4096,
            },
        }
    ]


def test_llm_post_processor_builds_exact_evaluated_messages_and_returns_cleaned_text():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {"content": "The report is ready. Please review it tomorrow."}
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        ),
    )

    raw_transcript = "  The report is ready. Please review it tomorrow.  "
    result = processor.process(raw_transcript)

    assert result == "The report is ready. Please review it tomorrow."
    messages = fake_client.calls[0]["messages"]
    assert hashlib.sha256(messages[0]["content"].encode()).hexdigest() == (
        "d4ffd45586d36678d6cbd8b2f58d32e7824b64da08b85b7d54839ca72fabb910"
    )
    assert messages[1:-1] == FEW_SHOT_MESSAGES
    assert messages[-1] == {
        "role": "user",
        "content": "The report is ready. Please review it tomorrow.",
    }


def test_llm_post_processor_appends_vocab_only_to_system_message():
    processor = LLMPostProcessor(
        client=object(), user_vocab={"q win": "Qwen", "murmer": "murmur"}
    )

    messages = processor.build_messages("  raw transcript  ")

    assert messages[0]["content"] == (
        f"{DEFAULT_SYSTEM_PROMPT}\n\nAdditional user vocabulary:\n"
        "- q win -> Qwen\n- murmer -> murmur"
    )
    assert messages[1:-1] == FEW_SHOT_MESSAGES
    assert messages[-1] == {"role": "user", "content": "raw transcript"}


def test_llm_post_processor_context_budget_boundary():
    max_tokens = 256
    at_limit = [{"role": "user", "content": "a" * ((3686 - max_tokens) * 4)}]
    over_limit = [{"role": "user", "content": at_limit[0]["content"] + "a"}]

    assert LLMPostProcessor._estimate_tokens(at_limit) == 3686 - max_tokens
    assert LLMPostProcessor._fits_context(at_limit, max_tokens) is True
    assert LLMPostProcessor._fits_context(over_limit, max_tokens) is False


def test_llm_post_processor_skips_oversize_input_without_calling_client(capsys):
    private_text = "private dictated phrase " * 1000
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process(private_text) == private_text.strip()
    assert fake_client.calls == []
    stdout = capsys.readouterr().out
    assert (
        "cleanup skipped because the estimated request exceeded the context budget"
        in stdout
    )
    assert "private dictated phrase" not in stdout


def test_llm_post_processor_returns_original_text_on_failure():
    class FailingClient:
        def chat(self, *args, **kwargs):
            raise RuntimeError("connection failed")

    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=FailingClient(),
        )
    )

    assert processor.process("keep this text") == "keep this text"


def test_llm_post_processor_failure_output_is_content_safe(capsys):
    class FailingClient:
        def chat(self, *args, **kwargs):
            raise RuntimeError("private dictated text leaked in exception")

    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=FailingClient(),
        )
    )

    assert processor.process("private dictated text") == "private dictated text"
    stdout = capsys.readouterr().out
    assert "Ollama post-processing failed" in stdout
    assert "private dictated text" not in stdout
    assert "leaked in exception" not in stdout


def test_llm_post_processor_normalizes_wrapped_output_before_accepting():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {"content": '```text\n"Hello, world."\n```'}
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process("hello world") == "Hello, world."


def test_llm_post_processor_rejects_assistant_preamble_output():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {
            "content": "Here's the cleaned transcript: Hello, world.",
        }
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process("Hello world.") == "Hello world."


def test_llm_post_processor_allows_transcript_text_with_meta_like_phrases():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {
            "content": "I corrected the deployment note as requested.",
        }
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert (
        processor.process("I corrected the deployment note as requested.")
        == "I corrected the deployment note as requested."
    )


def test_llm_post_processor_rejects_length_explosion_output():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {
            "content": (
                "Hello world. "
                "This transcript has been expanded with a long explanation that should not be accepted "
                "because it is much longer than the original text and clearly exceeds the allowed output size."
            ),
        }
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process("Hello world.") == "Hello world."


def test_llm_post_processor_rejects_chat_or_list_shaped_output():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {
            "content": "Assistant: Hello, world.\n- Fixed punctuation\n- Corrected capitalization",
        }
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process("Hello world.") == "Hello world."


def test_llm_post_processor_rejects_single_bullet_output():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {
        "message": {
            "content": "- Hello, world.",
        }
    }
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name=MODEL_NAME,
            client=fake_client,
        )
    )

    assert processor.process("Hello world.") == "Hello world."


def test_ollama_client_warm_loads_existing_model_without_generation():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.list_response = {"models": [{"name": MODEL_NAME}]}
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name=MODEL_NAME,
        client=fake_client,
    )

    warmed = client.warm(keep_alive="15m")

    assert warmed is True
    assert fake_client.calls == [
        {
            "model": MODEL_NAME,
            "messages": [],
            "stream": False,
            "keep_alive": "15m",
        }
    ]


def test_ollama_client_warm_skips_missing_model():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.list_response = {"models": [{"name": "other-model:latest"}]}
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name=MODEL_NAME,
        client=fake_client,
    )

    warmed = client.warm()

    assert warmed is False
    assert fake_client.calls == []


def test_check_ollama_connection_reports_available_model():
    class FakeClient:
        def __init__(self, endpoint, model_name, timeout):
            self.endpoint = endpoint
            self.model_name = model_name
            self.timeout = timeout

        def is_model_available(self):
            return True

    result = check_ollama_connection(
        endpoint=" http://localhost:11434 ",
        model_name=MODEL_NAME,
        timeout=2,
        client_factory=FakeClient,
    )

    assert result.ok is True
    assert MODEL_NAME in result.message


def test_check_ollama_connection_reports_missing_model():
    class FakeClient:
        def __init__(self, endpoint, model_name, timeout):
            pass

        def is_model_available(self):
            return False

    result = check_ollama_connection(
        endpoint="http://localhost:11434",
        model_name="missing:latest",
        client_factory=FakeClient,
    )

    assert result.ok is False
    assert "not installed" in result.message
