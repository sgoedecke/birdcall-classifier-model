## Data

https://huggingface.co/datasets/sgoedecke/powerful_owl_5s_16k

- download a bunch of audio files from yt/xeno-canto/wherever of the owl call you want
- convert them all to .wav and run `processing.py` in the directory, which will pull out the bird noises as segments
- then run `build-dataset.py` to stick them in a dataset on HF
- you may want to do some manual checks in the middle of that step

## Training

https://huggingface.co/sgoedecke/wav2vec2_owl_classifier_v3

- then run `birds.py` on a LambdaLabs or similar host to pull down wav2vec2, the larger birdcalls dataset, build a yes/no classifier from that plus your wav files, and finetune wav2vec2 on it
- That will upload a model to HF

## Deployment

- ??
- Throw a big file up, segment it into five second segments, drop segments with no noise, run the rest against the classifier, I guess

scp -i ~/.ssh/id_rsa ./owl-calls-yt.wav ubuntu@209.20.158.204:~/.
Then run `deployment.py`
