import os
import soundfile as sf
import librosa
from datasets import Dataset, Audio, Value, ClassLabel, Features, load_dataset, concatenate_datasets
import huggingface_hub

def create_dataset_from_folders(folders_labels):
    audio_paths = []
    labels = []
    filenames = []
    for folder, label in folders_labels:
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder, filename)
                audio_paths.append(file_path)
                labels.append(label)
                filenames.append(filename)
    return audio_paths, labels, filenames

def prepare_dataset(audio_paths, labels, filenames):
    # Convert lists to a HuggingFace Dataset
    data = {"audio": audio_paths, "label": labels, "filename": filenames}
    dataset = Dataset.from_dict(data)
    # Cast the 'audio' column to the 'Audio' feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # Cast the 'label' column to the 'ClassLabel' feature type
    dataset = dataset.cast_column("label", ClassLabel(names=["owl", "not_owl"]))
    # Cast the 'filename' column to the 'Value' feature type
    dataset = dataset.cast_column("filename", Value("string"))
    return dataset

# Login to Hugging Face Hub
huggingface_hub.login()

# Define your folders and labels
folders_labels = [
    ("dataset/owls", "owl"),
    ("dataset/not_owls", "not_owl")
]

# Load the existing dataset from the Hugging Face Hub
dataset_name = 'sgoedecke/powerful_owl_5s_16k'
new_dataset_name = 'sgoedecke/powerful_owl_5s_16k_v2'

existing_dataset = load_dataset(dataset_name, split='train')

# Create and prepare new data
audio_paths, labels, filenames = create_dataset_from_folders(folders_labels)
new_data = prepare_dataset(audio_paths, labels, filenames)

# Concatenate the new data with the existing dataset
updated_dataset = concatenate_datasets([existing_dataset, new_data])

# Push the updated dataset back to the Hugging Face Hub
updated_dataset.push_to_hub(new_dataset_name)

print(updated_dataset)
