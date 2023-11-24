import torch
from transformers import PerceiverForMultimodalAutoencoding
import numpy as np

import os


import scipy.io.wavfile

from utils import *



# TODO : load some video
video_names = list_ucf_videos()
video_path = fetch_ucf_video(video_names[0])

# Extract audio using FFMPEG and encode as pcm float wavfile (only format readable by scipy.io.wavfile).
import os
os.system(f"""ffmpeg -i "{video_path}"  -c copy  -f wav -map 0:a pcm_f32le -ar 48000 before.wav""") # TODO : Not that

sample_rate, audio = scipy.io.wavfile.read("before.wav")
if audio.dtype == np.int16:
  audio = audio.astype(np.float32) / 2**15
elif audio.dtype != np.float32:
  raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

video = load_video(video_path)
save_gif(video, path="before.gif")


#table([to_gif(video), play_audio(audio)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", low_cpu_mem_usage=False)
model.to(device)

reconstruction = autoencode_video(video[None, :16], audio[None, :16*AUDIO_SAMPLES_PER_FRAME, 0:1], model, device)

save_gif(reconstruction["image"][0].numpy(), path="after.gif")
save_audio(np.array(reconstruction["audio"][0].numpy(), path="after.wav"))

