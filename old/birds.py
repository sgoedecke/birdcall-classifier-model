# Fine-tuning wav2vec2 on the birdcall dataset to produce a birdcall classifier

# pip install soundfile librosa evaluate transformers
# pip install accelerate -U

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import evaluate
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.preprocessing import LabelEncoder

import huggingface_hub
huggingface_hub.login()


# birdcalls = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")

birdcalls_raw = load_dataset("tglcourse/5s_birdcall_samples_top20", cache_dir="~/sean-birds-testing-usw3")
birdcalls = birdcalls_raw["train"].train_test_split(test_size=0.5)

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


# Preprocess our dataset to ensure 16k sampling rate (should already be the case)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
birdcalls = birdcalls.cast_column("audio", Audio(sampling_rate=16_000))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate#, truncation=True#,max_length=16000
    )
    return inputs

birdcalls = birdcalls.map(preprocess_function, remove_columns="audio", batched=True)


# Build our model
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="wav2vec2_birdcalls",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    # gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
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

dataset = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))['test'].select(range(1000))
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

# from transformers import pipeline
# classifier = pipeline("audio-classification", model="sgoedecke/wav2vec2_birdcalls")

# def test_record(index):
#     audio_record = encoded_dataset[index]
#     print(classifier(audio_record['audio']))
#     print(id2label[audio_record['label']])


# test_record(1)

# This works on the LambdaLabs metal

# model.save_pretrained("wav2vec2_owl_classifier")
# feature_extractor.save_pretrained("wav2vec2_owl_classifier")

# repo = Repository("wav2vec2_owl_classifier", clone_from="sgoedecke/wav2vec2_owl_classifier")
# repo.git_add()
# repo.git_commit("Pushing model and feature extractor to hub")
# repo.git_push()