import nnabla as nn
import nnabla.functions as F
import numpy as np


def stft(x, window_size, stride, fft_size,
         window_type='hanning', center=True, pad_mode='reflect'):
    if window_type == 'hanning':
        window_func = np.hanning(window_size + 1)[:-1]
    elif window_type == 'hamming':
        window_func = np.hamming(window_size + 1)[:-1]
    elif window_type == 'rectangular' or window_type is None:
        window_func = np.ones(window_size)
    else:
        raise ValueError("Unknown window type {}.".format(window_type))

    # pad window if `fft_size > window_size`
    if fft_size > window_size:
        diff = fft_size - window_size
        window_func = np.pad(
            window_func, (diff // 2, diff - diff // 2), mode='constant')
    elif fft_size < window_size:
        raise ValueError(
            "FFT size has to be as least as large as window size.")

    # compute STFT filter coefficients
    mat_r = np.zeros((fft_size // 2 + 1, 1, fft_size))
    mat_i = np.zeros((fft_size // 2 + 1, 1, fft_size))

    for w in range(fft_size // 2 + 1):
        for t in range(fft_size):
            mat_r[w, 0, t] = np.cos(2. * np.pi * w * t / fft_size)
            mat_i[w, 0, t] = -np.sin(2. * np.pi * w * t / fft_size)

    conv_r = nn.Variable.from_numpy_array(mat_r * window_func)
    conv_i = nn.Variable.from_numpy_array(mat_i * window_func)

    if center:
        # pad at begin/end (per default this is a reflection padding)
        x = F.pad(x, (fft_size // 2, fft_size // 2), mode=pad_mode)

    # compute STFT
    y_r = F.convolution(x, conv_r, stride=(stride,))
    y_i = F.convolution(x, conv_i, stride=(stride,))

    return y_r, y_i
