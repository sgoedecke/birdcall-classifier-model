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
def handle_file(wav_file, index, total):
    print(f"Processing sample {index} out of {total}")
    wav_path = os.path.join(source_dir, wav_file)
    data, original_samplerate = sf.read(wav_path)
    while True:
        play_audio(data, original_samplerate)  # Use original samplerate for playback
        command = input("Press ENTER to move to /not_owls, 'o' then ENTER for /owls, 'r' to replay: ")
        
        if command == "o":
            copy_file(wav_path, os.path.join(owls_dir, wav_file))
            break
        elif command == "":
            copy_file(wav_path, os.path.join(not_owls_dir, wav_file))
            break
        elif command != "r":
            print("Invalid command. Skipping file.")
            break

# Get list of files to process
wav_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
total_files = len(wav_files)

# Iterate over all WAV files in the source directory
for index, wav_file in enumerate(wav_files, start=1):
    if os.path.exists(os.path.join(not_owls_dir, wav_file)) or os.path.exists(os.path.join(owls_dir, wav_file)):
        continue  # Skip this file if already processed
    handle_file(wav_file, index, total_files)
