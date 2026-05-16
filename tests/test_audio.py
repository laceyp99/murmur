import numpy as np

from src.audio import AudioRecorder


def test_audio_callback_hands_block_to_registered_callback():
    recorder = AudioRecorder.__new__(AudioRecorder)
    recorder._lock = __import__("threading").RLock()
    recorder._recording = True
    recorder._audio_data = []
    recorder._block_callback = None

    callback_blocks = []
    recorder.set_block_callback(callback_blocks.append)

    input_block = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    recorder._audio_callback(input_block, frames=3, time_info=None, status=None)

    assert len(recorder._audio_data) == 1
    np.testing.assert_array_equal(recorder._audio_data[0], input_block)
    assert len(callback_blocks) == 1
    np.testing.assert_array_equal(callback_blocks[0], input_block)