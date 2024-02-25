# run this against files downloaded from xeno-canto (or anywhere, I guess) to segment them into lots of little hoots
# I'm pretty sure this somehow provides duplicate data, but whatever - that saves me generating
# more synthetic data later, I suppose

# for f in *.mp3; do ffmpeg -i "$f" -acodec pcm_s16le -ar 16000 "${f%.mp3}.wav"; done
# for f in *.m4a; do ffmpeg -i "$f" -acodec pcm_s16le -ar 16000 "${f%.m4a}.wav"; done

# import os
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent

# # Ensure the output directory exists
# output_dir = "hoots"
# os.makedirs(output_dir, exist_ok=True)

# # Iterate over all WAV files in the current directory
# for wav_file in os.listdir('.'):
#     if wav_file.endswith(".wav"):
#         print(f"Processing {wav_file}...")
#         audio = AudioSegment.from_file(wav_file)        
#         # Detect non-silent chunks
#         nonsilent_chunks = detect_nonsilent(
#             audio,
#             min_silence_len=100,  # Minimum length of silence in milliseconds that separates chunks
#             silence_thresh=audio.dBFS + 5  # Adjust based on your audio
#         )        
#         buffer = 1000  # Buffer in milliseconds (1 second)        
#         # Process and export each detected chunk
#         for i, (start, end) in enumerate(nonsilent_chunks):
#             # Add buffer but ensure start and end are within audio bounds
#             start = max(0, start - buffer)
#             end = min(len(audio), end + buffer)            
#             segment = audio[start:end]
#             segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_hoot_{i}.wav"), format="wav")


# import os
# from pydub import AudioSegment

# # Ensure the output directory exists
# output_dir = "hoots"
# os.makedirs(output_dir, exist_ok=True)

# # Segment length and overlap in milliseconds
# segment_length = 5000  # 5 seconds
# overlap = 1000  # 1-second overlap on each side

# # Iterate over all WAV files in the current directory
# for wav_file in os.listdir('./data/inaturalist/'):
#     if wav_file.endswith(".wav"):
#         print(f"Processing {wav_file}...")
#         audio = AudioSegment.from_file('./data/inaturalist/' + wav_file)
#         # Calculate the average dBFS of the entire audio
#         average_dbfs = audio.dBFS
#         # Calculate number of segments to process, considering the overlap
#         num_segments = max(1, int(((len(audio) - overlap) / (segment_length - overlap))))
#         for i in range(num_segments):
#             # Calculate start and end considering the overlap, ensuring not to exceed audio length
#             start = i * (segment_length - overlap)
#             end = start + segment_length
#             end = min(end, len(audio))  # Ensure end does not exceed audio length
#             segment = audio[start:end]
#             # Calculate the maximum dBFS of the segment
#             max_segment_dbfs = max(segment[i:i+500].dBFS for i in range(0, len(segment), 500))  # Check max dBFS in 1-second intervals within the segment
#             # Define a threshold for considering a segment "silent" relative to the average dBFS of the entire file
#             # For example, if the segment's max dBFS is within 5 dB of the file's average dBFS, consider it silent
#             silence_threshold_db = 3
#             if (segment.dBFS - max_segment_dbfs) < silence_threshold_db:
#                 # If the segment is not considered "silent", export it
#                 segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_segment_{i}.wav"), format="wav")


# All this dBFS stuff didn't work well. I think this just needs manual curation

import os
from pydub import AudioSegment

# Ensure the output directory exists
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

# Define the segment length in milliseconds
segment_length = 5000  # 5 seconds

ddir = './data/youtube/'
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
    

