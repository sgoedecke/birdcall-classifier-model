import os
from pydub import AudioSegment

def parse_report(report_filename):
    import re
    segments = {}
    with open(report_filename, 'r') as file:
        lines = file.readlines()
    current_file = None
    for line in lines:
        if line.startswith('File:'):
             # Remove './data/' from the filename and strip any trailing whitespace or newline characters
             current_file = line.split(' ')[1].strip().replace('./data/', '')
             segments[current_file] = []
        elif 'Start' in line:
            times = re.findall(r'\d{2}:\d{2}:\d{2}', line)  # This will reliably extract HH:MM:SS formatted time strings
            if len(times) == 2:
                start_sec = sum(int(t) * 60**i for i, t in enumerate(reversed(times[0].split(':'))))
                end_sec = sum(int(t) * 60**i for i, t in enumerate(reversed(times[1].split(':'))))
                segments[current_file].append((start_sec * 1000, end_sec * 1000))  # Convert seconds to milliseconds
    return segments


# Path to the report file
report_filename = "owl_detection_report.txt"
# Read segments from report
file_segments = parse_report(report_filename)

# Directory setup
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

ddir = './data/'
# Process each WAV file mentioned in the report
for wav_file, segments in file_segments.items():
    full_path = os.path.join(ddir, wav_file)
    if os.path.exists(full_path):
        print(f"Processing {wav_file}...")
        audio = AudioSegment.from_file(full_path)
        for index, (start_ms, end_ms) in enumerate(segments):
            segment = audio[start_ms:end_ms]
            segment.export(os.path.join(output_dir, f"{wav_file[:-4]}_segment_{index}.wav"), format="wav")
    else:
        print(f"File not found: {full_path}")
