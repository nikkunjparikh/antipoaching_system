import tensorflow_hub as hub
import numpy as np
import librosa

# Load YAMNet model
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load an audio file (replace with your file)
waveform, sr = librosa.load("test_animal_sound.wav", sr=16000)  # YAMNet expects 16kHz

# Run inference
scores, embeddings, spectrogram = model(waveform)

# Find top 5 predictions
class_map_path = hub.resolve("https://tfhub.dev/google/yamnet/1").download_handler().get('yamnet_class_map.csv')
class_names = [line.strip() for line in open(class_map_path)][1:]

mean_scores = np.mean(scores, axis=0)
top5 = np.argsort(mean_scores)[::-1][:5]

print("\nTop predictions:")
for i in top5:
    print(class_names[i], mean_scores[i])
