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

