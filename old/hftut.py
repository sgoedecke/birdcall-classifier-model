# HF classifier tutorial

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import evaluate
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np

# pip install soundfile librosa evaluate transformers
# pip install accelerate -U
# import huggingface_hub
# huggingface_hub.login()

minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
# birdcalls = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")

minds = minds.train_test_split(test_size=0.2)

minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])

labels = minds["train"].features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="my_awesome_mind_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
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
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()


dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]

from transformers import pipeline

classifier = pipeline("audio-classification", model="sgoedecke/my_awesome_mind_model")
classifier(audio_file)

# [
#     {'score': 0.09766869246959686, 'label': 'cash_deposit'},
#     {'score': 0.07998877018690109, 'label': 'app_error'},
#     {'score': 0.0781070664525032, 'label': 'joint_account'},
#     {'score': 0.07667109370231628, 'label': 'pay_bill'},
#     {'score': 0.0755252093076706, 'label': 'balance'}
# ]

# This works on the LambaLabs metal