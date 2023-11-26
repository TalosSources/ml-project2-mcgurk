import torch
#from transformers import PerceiverForMultimodalAutoencoding
from models import PerceiverForMultimodalAutoencoding
import numpy as np

import os


import scipy.io.wavfile

from utils import *

import pickle
with open("video_autoencoding_checkpoint.pystate", "rb") as f:
  params = pickle.loads(f.read())
with open("params_print.txt", "w") as pp:
  pp.write(f"{params}")

video, audio = load_video_and_audio(0)


#table([to_gif(video), play_audio(audio)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
model.to(device)

sample_taken = 16
reconstruction = autoencode_video(video[None, :sample_taken], audio[None, :sample_taken*AUDIO_SAMPLES_PER_FRAME, 0:1], model, device)

save_gif(reconstruction["image"][0].numpy(), path="after.gif")
save_audio(np.array(reconstruction["audio"][0].numpy()), path="after.wav")

scores, indices = torch.topk(torch.softmax(reconstruction["label"], dim=1), k=5)
for score, index in zip(scores[0], indices[0]):
  print("%s: %s" % (model.config.id2label[index.item()], score.item()))

