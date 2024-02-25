import requests
import os
import json


# Assuming response is already loaded into a variable named `response`
# response = [{description: "", sounds: [{"file_url": "http://example.com/sound1.mp3"}, {"file_url": "http://example.com/sound2.mp3"}]}]
f = open('index.json')
response = f.read()
response = json.loads(response)['results']
urls = []

for item in response:
    if item['description'] and 'juvenile' in item['description'].lower():  # Case-insensitive check
        continue  # Skip this item
    if item['description'] and 'immature' in item['description'].lower():  # Case-insensitive check
        continue  # Skip this item
    for sound in item['sounds']:
        url = sound['file_url']
        urls.append(url)

        
        filename = url.split('/')[-1]  # Extracts filename from URL
        filename = filename.split('?')[0]  # Removes query parameters
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")

# Example usage
# download_files(response)
