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