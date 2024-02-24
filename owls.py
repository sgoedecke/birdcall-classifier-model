# Fine-tuning wav2vec2 on the birdcall dataset plus my owl dataset to produce a boolean "powerful owl or not" classifier

# pip install soundfile librosa evaluate transformers
# pip install accelerate -U

from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoFeatureExtractor, AutoConfig, AutoModelForSequenceClassification
import evaluate
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import huggingface_hub
huggingface_hub.login()


# birdcalls = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")

birdcalls_raw = load_dataset("tglcourse/5s_birdcall_samples_top20", cache_dir="~/sean-birds-testing-usw3")

def change_label_to_not_owl(example):
    example['label'] = "not-owl"
    return example

# Apply the function to each example in the dataset
birdcalls_raw['train'] = birdcalls_raw['train'].map(change_label_to_not_owl)
birdcalls = birdcalls_raw["train"].train_test_split(test_size=0.5)
birdcalls = birdcalls.cast_column("audio", Audio(sampling_rate=16_000))

owls_raw = load_dataset("sgoedecke/powerful-owl-birdcalls", cache_dir="~/sean-birds-testing-usw3")
owls_raw = owls_raw.cast_column("audio", Audio(sampling_rate=16_000))

# just not enough data points, make it so the model can't learn nothing is an owl
# going ham here so it's 50/50 - this will almost certainly overfit but at least it should do _something_
owls_raw['train'] = concatenate_datasets([owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train']])
owls_raw['train'] = concatenate_datasets([owls_raw['train'], owls_raw['train'], owls_raw['train'], owls_raw['train']])


owls = owls_raw['train'].train_test_split(test_size=0.5)

birdcalls['train'] = concatenate_datasets([birdcalls['train'], owls['train']])
birdcalls['test'] = concatenate_datasets([birdcalls['test'], owls['test']])

# Generate a dict between encoded labels and their text names
unique_labels = set(birdcalls["train"]["label"])
encoder = LabelEncoder()
encoder.fit(list(unique_labels))  # Fit once using all unique labels

# Transform the labels in both training and test datasets
encoded_train_labels = encoder.transform(birdcalls["train"]["label"])
encoded_test_labels = encoder.transform(birdcalls["test"]["label"])

# Update the dataset with encoded labels
birdcalls["train"] = birdcalls["train"].add_column("encoded_labels", encoded_train_labels)
birdcalls["test"] = birdcalls["test"].add_column("encoded_labels", encoded_test_labels)

# Generate label2id and id2label mappings
label2id = {label: i for i, label in enumerate(encoder.classes_)}
id2label = {i: label for i, label in enumerate(encoder.classes_)}

birdcalls['train'] = birdcalls['train'].remove_columns(['label']).rename_column("encoded_labels", "label")
birdcalls['test'] = birdcalls['test'].remove_columns(['label']).rename_column("encoded_labels", "label")

# set(birdcalls['train']['label'])


# Preprocess our dataset to ensure 16k sampling rate (should already be the case)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def preprocess_function(examples):
    target_length = 5 * feature_extractor.sampling_rate  # 5 seconds * sampling rate
    padded_audio_arrays = []
    for audio_array in examples["audio"]:
        current_length = len(audio_array["array"])
        if current_length < target_length:
            # Calculate the number of samples to pad
            padding_length = target_length - current_length
            # Pad with zeros (silence)
            padded_audio = np.pad(audio_array["array"], (0, padding_length), mode='constant')
        elif current_length > target_length:
            # Optionally truncate the clip to the target length
            padded_audio = audio_array["array"][:target_length]
        else:
            # No padding needed
            padded_audio = audio_array["array"]
        padded_audio_arrays.append(padded_audio)
    # Use the feature_extractor as before, now with padded_audio_arrays
    inputs = feature_extractor(padded_audio_arrays, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs

birdcalls = birdcalls.map(preprocess_function, remove_columns="audio", batched=True)


# Build our model
# accuracy = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     predictions = np.argmax(eval_pred.predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return metrics

num_labels = len(id2label)

# For some reason having this in meant I started training with 0.0 loss and it never changed
#config = AutoConfig.from_pretrained('facebook/wav2vec2-base', attention_dropout=0.1, hidden_dropout_prob=0.1)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label#, config=config
)

training_args = TrainingArguments(
    output_dir="wav2vec2_owl_classifier_v2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    # gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=True,
    resume_from_checkpoint=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=birdcalls["train"],
    eval_dataset=birdcalls["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---
# now let's test it
dataset = load_dataset("sgoedecke/powerful-owl-birdcalls", cache_dir="~/sean-birds-testing-usw3")
# dataset = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))['train'].select(range(20))
sampling_rate = dataset.features["audio"].sampling_rate
encoded_dataset = dataset.map(preprocess_function, batched=True)
def encode_labels(example):
    # Assuming 'label' is the key for your original string labels
    # Replace it with the actual key if different
    example["label"] = label2id[example["label"]]
    return example

# Apply the mapping to both the train and test datasets
encoded_dataset = encoded_dataset.map(encode_labels)

results = trainer.evaluate(encoded_dataset)
print(results)

# 3 epochs {'eval_loss': 2.7299187183380127, 'eval_accuracy': 0.22, 'eval_runtime': 4.189, 'eval_samples_per_second': 23.872, 'eval_steps_per_second': 0.955, 'epoch': 2.96}
# turning off gradient accumulation seems to have made a big improvement - the loss is already down below this before epoch 1 is over
# {'eval_loss': 2.4031777381896973, 'eval_accuracy': 0.41, 'eval_runtime': 4.1486, 'eval_samples_per_second': 24.105, 'eval_steps_per_second': 0.964, 'epoch': 10.0}
import soundfile as sf
import tempfile
from transformers import pipeline


classifier = pipeline("audio-classification", model="sgoedecke/wav2vec2_owl_classifier")


def save_array_to_temp_wav(audio_array, sampling_rate=16000):
    # Create a temporary file to save the audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    # Write the audio array to the file
    sf.write(temp_file.name, audio_array, sampling_rate)
    # Return the name of the temporary file
    return temp_file.name

def classify_audio_array(audio_array):
    # Save the audio array to a temporary WAV file
    temp_file_path = save_array_to_temp_wav(audio_array)
    # Classify the audio using the pipeline
    result = classifier(temp_file_path)
    # Optional: Clean up the temporary file if desired
    os.unlink(temp_file_path)
    return result

def test_record(index):
    audio_record = encoded_dataset[index]
    print(classify_audio_array(audio_record['audio']['array']))
    print(id2label[audio_record['label']])



test_record(1)
#==
import random

def inspect_dataset_samples(dataset, num_samples=5):
    # Randomly select indices for inspection
    indices = random.sample(range(len(dataset)), num_samples)
    for i, idx in enumerate(indices, 1):
        sample = dataset[idx]
        label = sample['label']
        input_values = np.array(sample['input_values'])
        print(f"Sample {i}:")
        print(f"Label: {label}")
        print(f"Input Values Shape: {input_values.shape}")
        # Optionally, display additional information about the input values
        print(f"Sample Input Values: {input_values[:10]}")  # Display first 10 values

