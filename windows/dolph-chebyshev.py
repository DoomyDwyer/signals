import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin

no_of_windows = 3
plot_no = 1
fig = plt.figure(1, figsize=(9.5, 7))

M_values = [31, 31, 101]
N_values = [50, 50, 120]
ripple_values = [-40, -200, -40]

fig.suptitle('Dolph-Chebyshev window')

for i in range(no_of_windows):
    M = M_values[i]
    N = N_values[i]
    ripple = ripple_values[i]

    hM1 = M // 2
    hM2 = M // 2 + 1
    hN = N // 2

    w = chebwin(M, ripple, M % 2 == 0)

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

    plt.subplot(no_of_windows, 2, plot_no)
    plt.plot(w)
    plt.title('w, M = %i, N = %i, ripple = %f' % (M, N, ripple))
    plot_no += 1

    plt.subplot(no_of_windows, 2, plot_no)
    mXdB = 20 * np.log10(mX)
    plt.plot(np.arange(-hN, hN) / float(N) * M, mXdB - max(mXdB))
    plt.title('W (dB), M = %i, N = %i, ripple = %f' % (M, N, ripple))
    plot_no += 1

plt.tight_layout()
plt.ion()
plt.show()
