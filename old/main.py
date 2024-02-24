# Initial attempt, cribbed from GPT4. Does not work.

import random
import logging

# import torch # used in the wav2vec stuff    
import tensorflow as tf

# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 1
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 32  # Batch-size for training and evaluating our model.
NUM_CLASSES = 10  # Number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 49
MAX_EPOCHS = 2  # Maximum number of training epochs.

MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Name of pretrained model from Hugging Face Model Hub

from datasets import load_dataset

# make sure we cache this in the storage dir that doesn't get nuked with the lambdalabs node
# https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20
# dataset = load_dataset("tglcourse/5s_birdcall_samples_top20", cache_dir="~/sean-birds-testing-usw")
split_dataset = load_dataset("sgoedecke/5s_birdcall_samples_16k", cache_dir="~/sean-birds-testing-usw3")

# Remove 'audio' and 'label' columns from the train split
train = split_dataset['train'].remove_columns(['audio', 'label'])

# Remove 'audio' and 'label' columns from the test split
test = split_dataset['test'].remove_columns(['audio', 'label'])


def dataset_generator(dataset):
    for i in range(len(dataset)):
        features = {key: dataset[key][i] for key in ['input_values', 'attention_mask']}
        label = dataset['labels'][i]
        yield (features, label)

output_signature = (
    {
        'input_values': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32)
    },
    tf.TensorSpec(shape=(), dtype=tf.int64)
)

tf_train_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train),
    output_signature=output_signature
).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)

tf_test_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(test),
    output_signature=output_signature
).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)

from transformers import TFWav2Vec2ForSequenceClassification, Wav2Vec2Config
import tensorflow as tf

# Load configuration for Wav2Vec2
config = Wav2Vec2Config.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_CLASSES)

# Load the pre-trained Wav2Vec2 model
model = TFWav2Vec2ForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config, from_pt=True)

# Ensure the model is compiled with the correct optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# for some reason the model has no idea what this optimizer is

# Train the model
model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=1)
