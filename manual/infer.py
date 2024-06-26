from pydub import AudioSegment
import os
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
import torch
import time
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python infer.py <paths_to_audio_files>")
    sys.exit(1)

def list_files(paths):
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            print(f"Warning: '{path}' is neither a file nor a directory.")
    return all_files

files = list_files(sys.argv[1:] )
print("Processing files: ", files)

# Load your model
print("Loading classifier model...")

# model_name = "sgoedecke/wav2vec2_owl_classifier_v3" # larger, slower
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # about 3x faster, smaller
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def look_for_owls(file):
    print(f"Segmenting {file} audio segment into 5 second chunks...")
    audio_segment = AudioSegment.from_file(file)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    num_elements_to_keep = len(audio) - (len(audio) % 80000)  # Trim to nearest 5 seconds
    audio = audio[:num_elements_to_keep]
    samples = audio.reshape(-1, 80000)  # Reshape samples into 5-second chunks
    # 7.5 seconds to calc logits for batch of 30 (5s chunks)
    # 26 seconds for a batch of 100 (5s chunks). Pretty static. All 29 cores are slammed.
    # The speed here is basically identical to my shitty DO node??
    batch_size = 100
    total_batches = len(samples) // batch_size + (1 if len(samples) % batch_size else 0)  # Calculate total number of batches

    print(f"Segmented into {len(samples)} chunks in {total_batches} batches...")
    results = []

    for batch_index in range(total_batches):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        inputs = processor(samples[start_index:end_index], sampling_rate=16000, return_tensors="pt", padding=True)
        print(f"Done preprocessing batch {batch_index} of {total_batches} for {file}. Calculating logits...")

        with torch.no_grad():  # Skip calculating gradients in the forward pass
            ts_start = time.time()
            logits = model(inputs.input_values.to(device)).logits
            ts_end = time.time()
            print(f"Processed {batch_size * 5} seconds of audio in {ts_end - ts_start} seconds.")
            found_owls = False
            for i, logit in enumerate(logits):
                label = "owl" if logit[0] > logit[1] else "not_owl"
                start_time = (batch_index * batch_size + i) * 5  # Adjusted start time calculation
                end_time = start_time + 5
                # Yield predictions for streaming
                if label == "owl":
                    found_owls = True
                    results.append((file, start_time, end_time, logit[0]))
                    print(f"Owl detected at {start_time} - {end_time} seconds!")
            if not found_owls:
                print(f"No owls detected in batch.")
    
    print(f"Done processing {file}!")
    return results

results = []
for file in files:
    results.append(look_for_owls(file))

print("Finished processing all files! ", results)
print("---------------------------------------------")
print("Some of these are likely to be false-positives. Please listen to the actual audio segments to confirm.")
print("Potential owls found:")

# Function to convert seconds to HH:MM:SS format
def seconds_to_hms(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = (seconds % 3600) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# Grouping timestamp tuples by filename
file_timestamps = defaultdict(list)
for sublist in results:
    for item in sublist:
        if len(item) == 0:
            next
        filename, start, end, logit = item
        file_timestamps[filename].append((start, end, logit))

# Printing timestamps grouped by filename, converted to HH:MM:SS
for filename, timestamps in file_timestamps.items():
    print(f"File: {filename}")
    for start, end, logit in timestamps:
        start_hms = seconds_to_hms(start)
        end_hms = seconds_to_hms(end)
        print(f"  Start: {start_hms}, End: {end_hms}, Chance: {logit:.2f}")

# Save report to a file
report_filename = "owl_detection_report_sewd.txt"
with open(report_filename, 'w') as file:
    file.write("Owl Detection Report\n")
    file.write("-------------------------------\n")
    for filename, timestamps in file_timestamps.items():
        file.write(f"File: {filename}\n")
        for start, end, logit in timestamps:
            start_hms = seconds_to_hms(start)
            end_hms = seconds_to_hms(end)
            file.write(f"  Start: {start_hms}, End: {end_hms}\n")
    file.write("-------------------------------\n")
    file.write("Note: Some detections may be false positives. Please verify by listening.\n")

print(f"Report saved to {report_filename}")