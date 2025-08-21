import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Create a window length
N = 256

# Generate window functions
windows = {
    "Rectangular": np.ones(N),
    "Hann": signal.windows.hann(N),
    "Hamming": signal.windows.hamming(N),
    "Blackman": signal.windows.blackman(N),
    "Bartlett": signal.windows.bartlett(N),
    "Kaiser (β=14)": signal.windows.kaiser(N, beta=14)
}

# Plot time-domain windows
plt.figure(figsize=(12, 8))
for i, (name, win) in enumerate(windows.items(), 1):
    plt.subplot(3, 2, i)
    plt.plot(win)
    plt.title(f"{name} Window")
    plt.grid(True)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
plt.tight_layout()
plt.suptitle("Time-Domain Representation of Window Functions", y=1.02, fontsize=16)
plt.show()
