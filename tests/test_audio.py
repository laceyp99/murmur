import numpy as np

from src.audio import AudioRecorder


def test_audio_callback_hands_block_to_registered_callback():
    recorder = AudioRecorder.__new__(AudioRecorder)
    recorder._lock = __import__("threading").RLock()
    recorder._recording = True
    recorder._audio_data = []
    recorder._block_callback = None
    recorder._on_block_callback_error = None
    recorder._block_callback_failed = False

    callback_blocks = []
    recorder.set_block_callback(callback_blocks.append)

    input_block = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    recorder._audio_callback(input_block, frames=3, time_info=None, status=None)

    assert len(recorder._audio_data) == 1
    np.testing.assert_array_equal(recorder._audio_data[0], input_block)
    assert len(callback_blocks) == 1
    np.testing.assert_array_equal(callback_blocks[0], input_block)


def test_audio_callback_logs_error_once_disables_callback_and_keeps_recording():
    recorder = AudioRecorder.__new__(AudioRecorder)
    recorder._lock = __import__("threading").RLock()
    recorder._recording = True
    recorder._audio_data = []
    recorder._block_callback = None
    recorder._on_block_callback_error = None
    recorder._block_callback_failed = False

    callback_errors = []
    callback_calls = []

    def failing_callback(block):
        callback_calls.append(block.copy())
        raise RuntimeError("vad callback broke")

    recorder.set_block_callback(failing_callback)
    recorder.set_block_callback_error_handler(callback_errors.append)

    first_block = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    second_block = np.array([[0.4], [0.5], [-0.6]], dtype=np.float32)

    recorder._audio_callback(first_block, frames=3, time_info=None, status=None)
    recorder._audio_callback(second_block, frames=3, time_info=None, status=None)

    assert len(recorder._audio_data) == 2
    np.testing.assert_array_equal(recorder._audio_data[0], first_block)
    np.testing.assert_array_equal(recorder._audio_data[1], second_block)
    assert len(callback_calls) == 1
    assert [str(error) for error in callback_errors] == ["vad callback broke"]
    assert recorder._block_callback is None
    assert recorder._block_callback_failed is True