import numpy as np
import pytest

from src.vad_audio import resample_audio


@pytest.mark.parametrize("source_rate", [22050, 24000, 32000, 44100, 48000, 96000])
@pytest.mark.parametrize("frequency", [440, 3000, 6000])
def test_downsampling_preserves_passband_amplitude(source_rate, frequency):
    positions = np.arange(source_rate) / source_rate
    audio = np.sin(2 * np.pi * frequency * positions).astype(np.float32)

    result = resample_audio(audio, source_rate, 16000)[160:-160]

    gain_db = 20 * np.log10(np.sqrt(2 * np.mean(result.astype(float) ** 2)))
    assert abs(gain_db) <= 0.1


@pytest.mark.parametrize("source_rate", [22050, 24000, 32000, 44100, 48000, 96000])
@pytest.mark.parametrize("frequency", [9000, 9200, 10500, 12000, 15000])
def test_downsampling_rejects_frequencies_above_target_nyquist(source_rate, frequency):
    if frequency >= source_rate / 2:
        pytest.skip("frequency is not representable at this source rate")
    positions = np.arange(source_rate) / source_rate
    audio = np.sin(2 * np.pi * frequency * positions).astype(np.float32)

    # Ignore the clip boundaries, where edge padding cannot extend the sine wave.
    result = resample_audio(audio, source_rate, 16000)[160:-160]

    output_rms = np.sqrt(np.mean(result.astype(float) ** 2))
    assert output_rms <= 10 ** (-40 / 20) / np.sqrt(2)


@pytest.mark.parametrize("length", [1, 2, 3, 17, 101])
@pytest.mark.parametrize("source_rate", [22050, 24000, 32000, 44100, 48000])
def test_downsampling_short_constant_audio_preserves_length_and_dc(source_rate, length):
    audio = np.full(length, 0.375, dtype=np.float32)
    original = audio.copy()

    result = resample_audio(audio, source_rate, 16000)

    assert result.shape == (max(1, round(length * 16000 / source_rate)),)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.375, atol=1e-6)
    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("source_rate", [16000, 48000])
def test_resampling_empty_audio_returns_independent_float32_array(source_rate):
    audio = np.array([], dtype=np.float64)

    result = resample_audio(audio, source_rate, 16000)

    assert result.shape == (0,)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, audio)


def test_same_rate_returns_independent_unchanged_audio():
    audio = np.array([0.2, -0.5, 0.7], dtype=np.float32)

    result = resample_audio(audio, 16000, 16000)

    np.testing.assert_array_equal(result, audio)
    assert not np.shares_memory(result, audio)


@pytest.mark.parametrize(
    ("source_rate", "target_rate"), [(0, 16000), (-1, 16000), (16000, 0), (16000, -1)]
)
def test_resampling_rejects_nonpositive_rates(source_rate, target_rate):
    with pytest.raises(ValueError, match="sample rates must be positive"):
        resample_audio(np.ones(16, dtype=np.float32), source_rate, target_rate)


def test_upsampling_retains_linear_interpolation():
    audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)

    result = resample_audio(audio, 8000, 16000)

    expected = np.interp(np.linspace(0, 2, 6), np.arange(3), audio)
    np.testing.assert_allclose(result, expected, atol=2e-7)
    assert result.dtype == np.float32


@pytest.mark.parametrize("source_rate", [22050, 24000, 32000, 44100, 48000])
def test_downsampling_centered_impulse_has_no_filter_delay(source_rate):
    audio = np.zeros(source_rate, dtype=np.float32)
    audio[source_rate // 2] = 1.0

    result = resample_audio(audio, source_rate, 16000)

    assert np.argmax(np.abs(result)) == 8000
    assert audio[source_rate // 2] == 1.0
    assert np.count_nonzero(audio) == 1
