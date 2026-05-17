from types import SimpleNamespace

from src.llm_postprocess import LLMPostProcessor, OllamaClient


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
        model_name="llama3.2:1b",
        timeout=15.0,
        client=fake_client,
    )

    result = client.generate("input text", max_tokens=42, temperature=0.0, system="system")

    assert result == "Cleaned output."
    assert fake_client.calls == [
        {
            "model": "llama3.2:1b",
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
        model_name="llama3.2:1b",
        client=fake_client,
    )

    assert client.generate("input text") == "Object response."


def test_ollama_client_chat_uses_messages_and_options():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {"message": {"content": "Cleaned output."}}
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name="llama3.2:1b",
        timeout=15.0,
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
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "input text"}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 42,
            },
        }
    ]


def test_llm_post_processor_builds_prompt_with_vocab_and_returns_cleaned_text():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {"message": {"content": "Quick recap: We met with Jane from Blue Ridge Data about the pilot. Jane asked if Noah can send the intake link and the Loom walkthrough by Friday."}}
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name="llama3.2:1b",
            client=fake_client,
        ),
        user_vocab={"q win": "Qwen", "murmer": "Murmur"},
    )

    result = processor.process("quick recap we met with jane from blue ridge data about the pilot the transcript may say brew ridge or blue rich but it should be blue ridge data jane asked if noah can send the intake link and the loom walkthrough by friday")

    assert result == "Quick recap: We met with Jane from Blue Ridge Data about the pilot. Jane asked if Noah can send the intake link and the Loom walkthrough by Friday."
    messages = fake_client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "Return only the cleaned transcript text." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Clean this transcript while preserving meaning and wording." in messages[1]["content"]
    assert "Return only the cleaned transcript text with no preamble or commentary." in messages[1]["content"]
    assert "Transcript:\nquick recap we met with jane from blue ridge data about the pilot the transcript may say brew ridge or blue rich but it should be blue ridge data jane asked if noah can send the intake link and the loom walkthrough by friday" in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": "Quick recap: We met with Jane from Blue Ridge Data about the pilot. Jane asked if Noah can send the intake link and the Loom walkthrough by Friday."}
    assert messages[3]["role"] == "user"
    assert "Preferred vocabulary and corrections:" in messages[3]["content"]
    assert "- q win -> Qwen" in messages[3]["content"]
    assert "- murmer -> Murmur" in messages[3]["content"]


def test_llm_post_processor_returns_original_text_on_failure():
    class FailingClient:
        def chat(self, *args, **kwargs):
            raise RuntimeError("connection failed")

    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name="llama3.2:1b",
            client=FailingClient(),
        )
    )

    assert processor.process("keep this text") == "keep this text"


def test_llm_post_processor_normalizes_wrapped_output_before_accepting():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.chat_response = {"message": {"content": "```text\n\"Hello, world.\"\n```"}}
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name="llama3.2:1b",
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
            model_name="llama3.2:1b",
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
            model_name="llama3.2:1b",
            client=fake_client,
        )
    )

    assert processor.process("I corrected the deployment note as requested.") == "I corrected the deployment note as requested."


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
            model_name="llama3.2:1b",
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
            model_name="llama3.2:1b",
            client=fake_client,
        )
    )

    assert processor.process("Hello world.") == "Hello world."


def test_ollama_client_warm_loads_existing_model_without_generation():
    fake_client = FakeOllamaPackageClient({"response": "ignored"})
    fake_client.list_response = {"models": [{"name": "llama3.2:1b"}]}
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name="llama3.2:1b",
        client=fake_client,
    )

    warmed = client.warm(keep_alive="15m")

    assert warmed is True
    assert fake_client.calls == [
        {
            "model": "llama3.2:1b",
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
        model_name="llama3.2:1b",
        client=fake_client,
    )

    warmed = client.warm()

    assert warmed is False
    assert fake_client.calls == []
