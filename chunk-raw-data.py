import os
from pydub import AudioSegment

# Ensure the output directory exists
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

# Define the segment length in milliseconds
segment_length = 5000  # 5 seconds

ddir = './data/'
# Iterate over all WAV files in the current directory
for wav_file in os.listdir(ddir):
    if wav_file.endswith(".wav"):
        print(f"Processing {wav_file}...")
        audio = AudioSegment.from_file(ddir + wav_file)
        if len(audio) >= segment_length:
            # Calculate number of full segments that can be created
            num_full_segments = len(audio) // segment_length
            for i in range(num_full_segments):
                start = i * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_segment_{i}.wav"), format="wav")
            # Handle the final segment, if any part remains
            if len(audio) % segment_length > 0:
                # Use the last 5 seconds as the final segment
                final_segment = audio[-segment_length:]
                final_segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_segment_{num_full_segments}.wav"), format="wav")
        else:
            silence_needed = segment_length - len(audio)
            silence_segment = AudioSegment.silent(duration=silence_needed)
            padded_audio = audio + silence_segment  # Concatenate audio with silence
            padded_audio.export(os.path.join(output_dir, f"{wav_file[:-4]}_segment_0.wav"), format="wav")
    

