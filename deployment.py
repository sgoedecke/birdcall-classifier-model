from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import pipeline
import os
import numpy as np

# scp -i ~/.ssh/id_rsa ./audio.wav ubuntu@instance-ip-address:~/.


audio = AudioSegment.from_file("owl-call-xc.wav")

nonsilent_segments = detect_nonsilent(
    audio,
    min_silence_len=500,
    silence_thresh=audio.dBFS + 3  # Adjust these parameters as needed
)

classifier = pipeline("audio-classification", model="sgoedecke/wav2vec2_owl_classifier_v2")


def pad_audio_segment(segment, desired_length_ms):
    # Calculate the padding length
    padding_length_ms = desired_length_ms - len(segment)
    if padding_length_ms > 0:
        # Create a silent segment for padding
        silent_segment = AudioSegment.silent(duration=padding_length_ms)
        # Pad the original segment with the silent segment
        padded_segment = segment + silent_segment
    else:
        # If the segment is longer than desired, truncate it
        padded_segment = segment[:desired_length_ms]
    return padded_segment


def classify_segment(audio, start, end, index, classifier):
    # Export the segment to a temporary file
    temp_file = f"temp_segment_{index}.wav"
    segment = audio[start:end]
    segment = pad_audio_segment(segment, 5000)
    segment.export(temp_file, format="wav")
    # Classify the segment using the audio classification pipeline
    result = classifier(temp_file)
    # Remove the temporary file to clean up
    os.remove(temp_file)
    return result

# Classify each non-silent segment
classified_segments = []
for i, (start, end) in enumerate(nonsilent_segments):
    classification_result = classify_segment(audio, start, end, i, classifier)
    classified_segments.append(classification_result)

# Example output
for i, result in enumerate(classified_segments):
    print(f"Segment {i+1}: {result}")

# it does work on the training data set!
# >>> classifier(np.array(owls[0]['input_values']))
# [{'score': 0.9999967813491821, 'label': 'owl'}, {'score': 3.2106850085256156e-06, 'label': 'not-owl'}]
# >>> classifier(np.array(owls[1]['input_values']))
# [{'score': 0.9999967813491821, 'label': 'owl'}, {'score': 3.212433966837125e-06, 'label': 'not-owl'}]