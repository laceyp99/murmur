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


def test_llm_post_processor_builds_prompt_with_vocab_and_returns_cleaned_text():
    fake_client = FakeOllamaPackageClient({"response": "Hello, world."})
    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name="llama3.2:1b",
            client=fake_client,
        ),
        user_vocab={"q win": "Qwen", "murmer": "Murmur"},
    )

    result = processor.process("hello world")

    assert result == "Hello, world."
    prompt = fake_client.calls[0]["prompt"]
    assert "hello world" in prompt
    assert "Preferred vocabulary and corrections:" in prompt
    assert "- q win -> Qwen" in prompt
    assert "- murmer -> Murmur" in prompt


def test_llm_post_processor_returns_original_text_on_failure():
    class FailingClient:
        def generate(self, **kwargs):
            raise RuntimeError("connection failed")

    processor = LLMPostProcessor(
        client=OllamaClient(
            endpoint="http://localhost:11434",
            model_name="llama3.2:1b",
            client=FailingClient(),
        )
    )

    assert processor.process("keep this text") == "keep this text"


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
