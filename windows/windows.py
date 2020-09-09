import numpy as np
from scipy.fft import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt

M = 51
hM1 = M // 2
hM2 = M // 2 + 1

N = 140
hN = N // 2

windows = ['boxcar', 'triang', 'hamming', 'hann', 'blackman', 'blackmanharris']
fig_no = 1

for window in windows:
    w = get_window(window, M, M % 2 == 0)

    fftbuffer = np.zeros(N)
    fftbuffer[:hM1] = w[hM2:]
    fftbuffer[N - hM2:] = w[:hM2]

    W = fft(fftbuffer)

    mX = np.zeros(N)
    mX[:hN] = np.abs(W)[hN:]
    mX[hN:] = np.abs(W)[:hN]

    pX = np.zeros(N)
    pX[:hN] = np.angle(W)[hN:]
    pX[hN:] = np.angle(W)[:hN]

    plt.figure(fig_no, figsize=(9.5, 7))

    plt.subplot(5, 1, 1)
    plt.plot(w)
    plt.title('%s window, M = %i' % (window, M))

    plt.subplot(5, 1, 2)
    plt.plot(fftbuffer)
    plt.title('fftbuffer, N = %i' % N)

    plt.subplot(5, 1, 3)
    plt.plot(np.arange(-hN, hN) / float(N) * M, mX-max(mX))
    plt.title('W (real part)')

    plt.subplot(5, 1, 4)
    mXdB = 20 * np.log10(mX)
    plt.plot(np.arange(-hN, hN) / float(N) * M, mXdB - max(mXdB))
    plt.title('W (real part - dB)')

    plt.subplot(5, 1, 5)
    plt.plot(np.arange(-hN, hN) / float(N) * M, pX)
    plt.title('W phase')

    plt.tight_layout()
    plt.ion()
    plt.show()

    fig_no += 1
