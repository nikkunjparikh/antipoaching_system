import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
file = "CatPain.wav"
y, sr = librosa.load(file, sr=16000)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC shape:", mfcc.shape)
print(mfcc)

# Plot the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(label='MFCC Coefficients')
plt.title('MFCC')
plt.tight_layout()
plt.show()
