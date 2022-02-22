import librosa as lr
import nnabla as nn
import nnabla.functions as F
import numpy as np
import pyworld
from librosa.filters import mel as librosa_mel_fn

from .helper import stft


def spectrogram(wave, window_size):
    r"""Computes the spectrogram from the waveform.

    Args:
        wave (nn.Variable): Input waveform of shape (B, 1, L).
        window_size (int): Window size.

    Returns:
        nn.Variable: The square spectrogram.
    """
    re, im = stft(wave, window_size=window_size,
                  stride=window_size // 4, fft_size=window_size)
    return F.pow_scalar(re**2 + im**2, 0.5)


def mel_spectrogram(wave, sr, window_size, n_mels=80, fmin=0, fmax=8000):
    r"""Returns mel-scale spectrogram.

    Args:
        wave (nn.Variable): Input waveform of shape (B, 1, L).
        sr (int): Sampling rate.
        window_size (int): Window size.
        n_mels (int): Number of mel banks.

    Returns:
        nn.Variable: Mel-scale spectrogram.
    """
    linear = spectrogram(wave, window_size)
    mel_basis = librosa_mel_fn(
        sr, window_size, n_mels=n_mels,
        fmin=fmin, fmax=fmax
    )
    basis = nn.Variable.from_numpy_array(mel_basis[None, ...])
    mels = F.batch_matmul(basis, linear)

    return mels


def compute_mel_spectrogram(wave, mel_basis, n_fft, hop_length, win_length):
    r"""Compute mel spectrogram using librosa.

    Args:
        wave (np.ndarray): Input waveform.
        mel_basis (np.ndarray) : Mel filter banks.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        win_length (int): Window length.

    Returns:
        np.ndarray: Mel-spectrogram array of shape (n_mel, T)
        np.ndarray: Energy array of shape (T,).
    """
    linear = lr.stft(wave, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length)
    linear = np.abs(linear)
    mel = np.dot(mel_basis, linear)
    energy = np.linalg.norm(linear, axis=0)

    return mel, energy


def compute_fundamental_frequency(wave, sr, hop_length):
    r"""Computes the fundamental frequency.

    Args:
        wave (np.ndarray): Input waveform audio.
        sr (int): Sampling rate.
        hop_length (int): Hop length.

    Returns:
        np.ndarray: Output pitch.
    """
    wave = wave.astype(np.float64)
    pitch, t = pyworld.dio(wave, sr, frame_period=hop_length / sr * 1000)
    pitch = pyworld.stonemask(wave, pitch, t, sr)

    return pitch
