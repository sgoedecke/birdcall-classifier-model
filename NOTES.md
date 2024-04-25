I'm trying to build a classifier model for the powerful owl

AFAICT, there aren't many audio models compared to visual/text. Facebook wav2vec2 is the best open model in town.

It looks like the main strategy is to fine-tune wav2vec2 on a dataset of bird calls

Pretty easy to rent compute from LambdaLabs, although I had to bounce around a few hosts before I found one that wasn't busted (wonky CUDA setup, etc). Maybe I could have fixed it with config but the utah one worked out of the box.
  Attaching filesystems makes it easy to turn the host off overnight, but forces you to stay in the same region, which makes sense. Mostly for caching anyway at this scale
  The HuggingFace way seems to be to use their backend as storage for your datasets/models and keep pushing up. I have no idea how the economics works out for them, it must be expensive as hell to store everyone's 6GB parquet and model files for all their messing-around projects for free

Juggling TF/Keras/transformers frameworks is very awkward. Easy to combine them by accident and try push one framework's object into another, with disastrous results. But there's a ton of machinery around HuggingFace that makes it easy once you do line it up.

Audio models operate on a large tensor of floats (ofc). Crucial to get your data sampling rate (khz) to match wav2vec2's sampling rate of 16khz. Lots of machinery to help with this too.

So far not a lot of luck - the training completes, but predictions suck. Loss is still at 2.x

Removing `gradient_accumulation_steps` seemed to have a really substantial impact. Loss is already ~1.5 at 3 epochs. 7 to go.

No great success yet, but this was surprisingly easy to get started. If it ends up being ~3 days from "never installed tensorflow" to "fine-tuned wav2vec2 on birdcalls" I'll be impressed with the LambdaLabs/HuggingFace/transformers industrial complex.

OK, 10 epochs worked! 41% accuracy, but spot checks were 100%

Next steps: adapt this one into a binary classifier for just owls

- get a lot of hoots
- turn those hoots into an actual dataset

OK, I think I accidentally trained the model to just say nothing was an owl. I think I need more owls in the dataset. 

Adding more owls doesn't seem to have fixed it, which is interesting. How can the loss/learning rate be so low when clearly it cant be getting that much right from the test set (if it never gets the owls right)?


random guess... maybe the trainer is somehow picking up from where it left off? the loss seems really low to start with (0.6949), so maybe I'm just trying to fine tune an already-busted model. I should delete the directory and see what happens
    I restarted it with the cache dir deleted and the starting loss was 0.0058. what.


Maybe I was screwing up my testing. This seems to indicate it does actually predict OK:
trainer.predict(birdcalls['test'].shuffle(seed=42).select(range(20)))
Probably the next step is figuring out why it's overfitting, which it almost certainly is. Some combination of need-more-owl-sounds and harvesting 5s clips that match the birdcall set (so the model can't just look for padding)
Yeah, it does predict fine on the training data. I couldn't make it predict by just adding padding, which suggests I do straight up need more owl sounds to make it work.


To try:
- just segment powerful-owl recordings into 5-second chunks naively
- instead of other birdcalls, train against background nature sounds as well (but there's still a size issue)
- try to crack the "you don't have a lot of owl data" issue (training on asymmetrical datasets) instead of working around it

OK, I've pulled a ton of data from inat and other sources. 1809 samples, from which I hope to filter at least 800 powerful owl calls. Filtering with python hasn't been successful. I'm going to chunk them into 5 second segments, then listen to them all and manually move them into `/hoots` if they contain owl calls, and then I'll upload the dataset as powerful-owl-5s or something. I should move the other sounds into `/not-hoots`. (I ended up making a script to autoplay and wait for input here)


Wow, that took a long time, even with my helper script! Like three full hours of classifying. Lessons: put a beep between sounds, if possible make it so that when you classify as owl it skips forward immediately, re-runnability is essential, replay-sound is essential.

Some long-range thoughts as the model trains: I should probably figure out how to bias against false positives here - if an owl calls once, it'll call a few times, so as long as my hit rate is 80%+ and I never false-pos I should be OK.
  i.e. I want high precision and I'm willing to trade low recall to do it. I should consider training with the F0.5 score as my key metric instead of the F1 score

Seems to work on the training data, at least! Tested this one out on the HF inference API as the model trained (it's at like epoch 6.5, down to .16 loss)

I'll train a fbeta model at the same time and then shut down my LambdaLabs node and do my testing on the HF inference API.

Well, it works! But kinda slow on my DO droplet

Batch
15 sec for a 1mb file
1 min for a 3mb file

Stream is broken

Piecwise
1.1 min for a 3mb file (3m40s)
On hugging face API, 43 sec after warmup

I should get streaming working. But I should also try to speed this up. Could at minimum quantize. Could even maybe run it with llama.cpp

parallelize the HF calls for sure

Struggling to quantize - and also I feel like this 300MB w2v2 model was already quantized, so maybe unnecessary

Student model? Seems pretty fiddly: https://medium.com/georgian-impact-blog/compressing-wav2vec-2-0-f41166e82dc2

Probably worth trying harder to distribute inference across CPU cores

I wonder if I can just straight up try larger chunks into the model, even though it was trained on smaller ones.
https://huggingface.co/blog/asr-chunking
`output = pipe("very_long_file.mp3", chunk_length_s=10, stride_length_s=(4, 2))`

yeah, larger chunks work fine. They seem to predict acceptably as well, which is an interesting bonus

Batching across cores doesn't do anything, the pipeline class already does spreads work across available CPUs. That's why it's so hard to bring the time down below 18-20 seconds for a 2 min file - I'm fighting an already-optimized thing. 

https://arxiv.org/pdf/2202.05993.pdf suggests that when you're running wav2vec2 on cheap GPUs for transcription, if you can transcribe as fast as audio comes in you're doing very well. Classifying should be much faster, but if it's in the same ballpark maybe 10s/min isn't so bad.

Even aggressive pruning of wav2vec2 only gives you like a 30% improvement https://www.researchgate.net/profile/Oswaldo-Ludwig/publication/374109762_Presentation_compressed_W2Vpdf/data/650db154c05e6d1b1c2745eb/Presentation-compressed-W2V.pdf?origin=publication_list 

I could try a performance-optimized smaller version of wav2vec, that claims to be almost 2x faster: https://huggingface.co/docs/transformers/model_doc/sew-d 

https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer - this is fascinating, and basically does what I suggested doing in my notes (using a visual model on spectrograms)


Trying a tiny sew-d model - the loss seems pretty static at 0.1/2. If that represents a 3-6x improvement from random guessing, maybe that's still worthwhile? Oh actually, screw loss - the f1 score is great. I think maybe we're good? 
The final model is 1/4 the size lol.

Wow, batched inference - which did not buy anything in wav2vec2 gives another 2x speedup on the sew-d model. Maybe we were bottlenecking on memory.

OK, today I'm going to try and (a) run some AO recordings through the website (https://data.acousticobservatory.org/projects/1/regions/26/audio_recordings/download) and (b) write up instructions to inference large files on server, which will probably mean a new repo 

on the website, 156MB FLAC: the upload OOMd, I think. That's kind of weird. 
Actually, increasing the timeout seems to have fixed it??? Maybe I was misreading dmesg.

OK, so the server now works on a 2H FLAC (processing time 7 min ish?). That's exciting. I want to see how fast it can rip through on a H100. CPU bound, even with 20 cores it doesn't make a difference. But putting it on the GPUs gives 3 OOMs speedup. (even the wav2vec2 model gets the same, though that's 3x slower than the SEW-D model)

---

TODO: improve dataset with new data, crack down on false positives. First step: AI-assisted chunking. Trust negatives, double-check positives.

1. Create a new working folder and go into it
2. Copy your wavs into a `./data/` folder
3. `python3 /Users/sgoedecke/Code/birds/server/manual/infer.py ./data/*.wav` to generate a report identifying the owl sounds from the wavs (will contain false positives, likely). This will generate the report in `owl_detection_report.txt`
4. `python3 /Users/sgoedecke/Code/birds/assisted-dataset-construction/chunk-raw-data.py` to fill a `/segments` folder with those 5 second chunks
5. `python3 /Users/sgoedecke/Code/birds/manual-audio-classifier.py` to go listen to those segments and stick them in owls/not-owls folders
6. `python3 /Users/sgoedecke/Code/birds/assisted-dataset-construction/update-dataset-with-new-classified-audio.py` to take those owls/not-owls snippets and supplement the existing powerful owl dataset (labelled `sgoedecke/powerful_owl_5s_16k_v2`)


20:43-51 I heard a hoot (1243s on)
TODO: try generate report with the SEW-d and see if it's different - it is very minor.
Note: sewd and the regular w2v2 model mostly overlap, but there are some small differences. Certainly w2v2 doesn't seem necessarily better.

TODO: pull data from acoustic observatory to catch more false positives

Note: when processing a bunch of what you expect are false positives, it's easier to just play them all in VLC than to use the script

f1 isn't gonna drop below .94 in either model at ~2k samples in the dataset