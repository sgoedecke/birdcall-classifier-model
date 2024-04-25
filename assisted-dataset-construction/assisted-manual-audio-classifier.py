import os
import shutil
import soundfile as sf
import simpleaudio as sa
import numpy as np

# Directories
source_dir = "./segments"
not_owls_dir = "./dataset/not_owls"
owls_dir = "./dataset/owls"

# Ensure target directories exist
os.makedirs(not_owls_dir, exist_ok=True)
os.makedirs(owls_dir, exist_ok=True)

def parse_report(report_filename):
    import re
    segments = {}
    with open(report_filename, 'r') as file:
        content = file.readlines()
    current_file = None
    for line in content:
        if line.startswith('File:'):
            current_file = line.strip().split(' ')[1]
            segments[current_file] = []
        elif 'Start' in line:
            times = re.findall(r'\d\d:\d\d:\d\d', line)
            start_time = sum(x * int(t) for x, t in zip([3600, 60, 1], times[0].split(':')))
            end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], times[1].split(':')))
            segments[current_file].append((start_time, end_time))
    return segments


# Function to change the speed of an audio file
def change_speed(data, samplerate, speed=1.0):
    indices = np.round(np.arange(0, len(data), speed))
    indices = indices[indices < len(data)].astype(int)
    return data[indices]

# Function to play audio from numpy array
def play_audio(data, samplerate):
    audio = data * (2**15 - 1) / np.max(np.abs(data))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, samplerate)
    play_obj.wait_done()

# Function to copy file based on user input
def copy_file(file_path, destination):
    shutil.copy(file_path, destination)
    print(f"File copied to {destination}")

# Adjusted function to handle file playback and user commands
def handle_file(wav_path, segments, speed=1.0):
    data, samplerate = sf.read(wav_path)
    for start, end in segments:
        segment_data = data[int(start * samplerate):int(end * samplerate)]
        while True:
            play_audio(segment_data, samplerate)
            command = input("Press ENTER to move to /not_owls, 'o' then ENTER for /owls, 'r' to replay: ")
            if command == "o":
                copy_file(wav_path, os.path.join(owls_dir, os.path.basename(wav_path)))
                break
            elif command == "":
                copy_file(wav_path, os.path.join(not_owls_dir, os.path.basename(wav_path)))
                break
            elif command != "r":
                print("Invalid command. Skipping file.")
                break

report_filename = "owl_detection_report.txt"
segments = parse_report(report_filename)

print(segments)

# Process files based on segments
for wav_path in segments:
    if os.path.exists(wav_path):
        handle_file(wav_path, segments[wav_path])
    else:
        print("Path not found:", wav_path)