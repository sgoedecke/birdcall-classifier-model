import librosa
from datasets import Dataset
import numpy as np
import os

import huggingface_hub
huggingface_hub.login()

def load_audio_files(directory, label=None):
    audio_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            audio, sampling_rate = librosa.load(path, sr=None)
            audio_data.append({"audio": {"array": audio, "sampling_rate": sampling_rate}, "label": label})
    return audio_data

# Example usage:
directory = "./hoots"  # Update this path
audio_data = load_audio_files(directory, label="owl")

# Create a Hugging Face Dataset from the audio data
dataset = Dataset.from_dict({"audio": [x['audio'] for x in audio_data], "label": [x['label'] for x in audio_data]})

# Optionally, split the dataset into train/test splits if needed
# dataset = dataset.train_test_split(test_size=0.2)

print(dataset)
print(dataset[0]['audio'])

dataset.push_to_hub("powerful-owl-birdcalls")
