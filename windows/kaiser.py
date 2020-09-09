import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal.windows import kaiser

alpha_max = 5
plot_no = 1
fig = plt.figure(1, figsize=(9.5, 7))

M = 21
hM1 = M // 2
hM2 = M // 2 + 1

N = 40
hN = N // 2

fig.suptitle('Kaiser window, M = %i, fft size = %i' % (M, N))

for alpha in range(alpha_max + 1):
    beta = alpha * np.pi
    w = kaiser(M, beta, M % 2 == 0)

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

    plt.subplot(alpha_max + 1, 2, plot_no)
    plt.plot(w)
    plt.title('w, alpha = %i, beta = %f' % (alpha, beta))
    plot_no += 1

    plt.subplot(alpha_max + 1, 2, plot_no)
    mXdB = 20 * np.log10(mX)
    plt.plot(np.arange(-hN, hN) / float(N) * M, mXdB - max(mXdB))
    plt.title('W (dB), alpha = %i, beta = %f' % (alpha, beta))
    plot_no += 1

plt.tight_layout()
plt.ion()
plt.show()
