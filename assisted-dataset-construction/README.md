This assumes you already have a dataset at `sgoedecke/powerful_owl_5s_16k` and a model to go with it. Here's how you use that model to classify new data for the dataset without having to manually go through thousands of clips.

1. Create a new working folder and go into it
2. Copy your wavs into a `./data/` folder
3. `python3 /Users/sgoedecke/Code/birds/server/manual/infer.py ./data/*.wav` to generate a report identifying the owl sounds from the wavs (will contain false positives, likely). This will generate the report in `owl_detection_report.txt`
4. `python3 /Users/sgoedecke/Code/birds/assisted-dataset-construction/chunk-raw-data.py` to fill a `/segments` folder with those 5 second chunks
5. `python3 /Users/sgoedecke/Code/birds/manual-audio-classifier.py` to go listen to those segments and stick them in owls/not-owls folders
6. `python3 /Users/sgoedecke/Code/birds/assisted-dataset-construction/update-dataset-with-new-classified-audio.py` to take those owls/not-owls snippets and supplement the existing powerful owl dataset (labelled `sgoedecke/powerful_owl_5s_16k_v2`)

