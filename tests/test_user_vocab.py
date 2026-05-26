import json

from src.user_vocab import load_user_vocab, save_user_vocab


def test_load_user_vocab_returns_empty_mapping_when_missing(tmp_path):
    vocab_path = tmp_path / "user_vocab.json"

    assert load_user_vocab(vocab_path) == {}


def test_save_user_vocab_round_trips_json_mapping(tmp_path):
    vocab_path = tmp_path / "user_vocab.json"
    vocab = {
        "brew ridge": "Blue Ridge Data",
        "murmer": "Murmur",
    }

    save_user_vocab(vocab, vocab_path)

    assert load_user_vocab(vocab_path) == vocab
    assert json.loads(vocab_path.read_text(encoding="utf-8")) == vocab


def test_load_user_vocab_returns_empty_mapping_for_invalid_json(tmp_path):
    vocab_path = tmp_path / "user_vocab.json"
    vocab_path.write_text("not json", encoding="utf-8")

    assert load_user_vocab(vocab_path) == {}