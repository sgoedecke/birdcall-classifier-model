import json
import requests
from pydub import AudioSegment

headers = {"Authorization": f"Bearer hf_key"}
API_URL = "https://api-inference.huggingface.co/models/sgoedecke/wav2vec2_owl_classifier_v3"

def query(audio_bytes):
    response = requests.request("POST", API_URL, headers=headers, data=audio_bytes)
    return json.loads(response.content.decode("utf-8"))

def chunk_and_query(filename, chunk_length_ms=5000):
    # Load the audio file
    audio = AudioSegment.from_file(filename)
    # Calculate the number of chunks
    num_chunks = len(audio) // chunk_length_ms
    results = []
    for i in range(num_chunks):
        # Calculate the start and end of this chunk
        start_ms = i * chunk_length_ms
        end_ms = start_ms + chunk_length_ms
        # Extract the chunk
        chunk = audio[start_ms:end_ms]
        # Export the chunk to bytes
        chunk_bytes = chunk.export(format="wav")
        # Send the chunk to the API and store the response
        result = query(chunk_bytes.read())
        results.append(result)
        # Close the BytesIO object
        chunk_bytes.close()
    return results

# Example usage
filename = "behavior-of-aus-powerful-owl.wav"

results = chunk_and_query(filename)
for result in results:
    print(result)

chunk_length_seconds = 5  # Duration of each audio chunk

for i, result in enumerate(results):
    start_time = i * chunk_length_seconds
    end_time = start_time + chunk_length_seconds
    
    # Check if the result is a successful prediction and corresponds to an owl sound
    if isinstance(result, list) and result[0]['label'] == 'owl':
        print(f"Owl sound detected from {start_time} to {end_time} seconds.")