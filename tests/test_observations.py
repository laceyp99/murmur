import pytest

from src.observations import emit_observation, observe_transcription


def emit():
    emit_observation(
        outcome="success", seconds=0.25, code="whisper_completed", category="success"
    )


def test_observation_scopes_restore_observer_and_stage_after_failure():
    outer, inner = [], []

    def fail_in_scope():
        with observe_transcription(inner.append, stage="whisper_live"):
            emit()
            raise ValueError("private text")

    with observe_transcription(outer.append):
        emit()
        with pytest.raises(ValueError, match="private text"):
            fail_in_scope()
        emit()
    emit()
    assert len(outer) == 2
    assert outer[0] == outer[1]
    assert outer[0]["stage"] == "whisper_offline"
    assert inner[0]["stage"] == "whisper_live"
    assert set(inner[0]) == {
        "stage",
        "outcome",
        "seconds",
        "code",
        "category",
        "metadata",
    }
    assert inner[0]["metadata"] == {"whisper_model": None, "device": None}


@pytest.mark.parametrize("error", [RuntimeError("private text"), SystemExit(1)])
def test_callback_failures_are_fully_contained(error):
    def broken(event):
        raise error

    with observe_transcription(broken):
        emit()
