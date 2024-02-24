# run this against files downloaded from xeno-canto (or anywhere, I guess) to segment them into lots of little hoots
# I'm pretty sure this somehow provides duplicate data, but whatever - that saves me generating
# more synthetic data later, I suppose

import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Ensure the output directory exists
output_dir = "hoots"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all WAV files in the current directory
for wav_file in os.listdir('.'):
    if wav_file.endswith(".wav"):
        print(f"Processing {wav_file}...")
        audio = AudioSegment.from_file(wav_file)        
        # Detect non-silent chunks
        nonsilent_chunks = detect_nonsilent(
            audio,
            min_silence_len=100,  # Minimum length of silence in milliseconds that separates chunks
            silence_thresh=audio.dBFS + 5  # Adjust based on your audio
        )        
        buffer = 1000  # Buffer in milliseconds (1 second)        
        # Process and export each detected chunk
        for i, (start, end) in enumerate(nonsilent_chunks):
            # Add buffer but ensure start and end are within audio bounds
            start = max(0, start - buffer)
            end = min(len(audio), end + buffer)            
            segment = audio[start:end]
            segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_hoot_{i}.wav"), format="wav")
