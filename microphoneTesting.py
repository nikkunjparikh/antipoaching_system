import sounddevice as sd
import numpy as np
import wavio

# Settings
duration = 5  # seconds
sample_rate = 16000

print(" Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print(" Recording complete.")

# Save as WAV
wavio.write("test_recording.wav", audio, sample_rate, sampwidth=2)
print("File saved as test_recording.wav")
