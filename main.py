import torch
from transformers import PerceiverForMultimodalAutoencoding
from PIL import Image
import tqdm
import numpy as np


# CONSTANTS? TODO
SAMPLES_PER_PATCH = 16
AUDIO_SAMPLES_PER_FRAME = 48000 // 25

def autoencode_video(images, audio):
  
  # only create entire video once as inputs
  inputs = {'image': torch.from_numpy(np.moveaxis(images, -1, 2)).float().to(device),
          'audio': torch.from_numpy(audio).to(device),
          'label': torch.zeros((images.shape[0], 700)).to(device)}
  
  nchunks = 128
  reconstruction = {}
  for chunk_idx in tqdm(range(nchunks)):
        image_chunk_size = np.prod(images.shape[1:-1]) // nchunks
        audio_chunk_size = audio.shape[1] // SAMPLES_PER_PATCH // nchunks
        subsampling = {
            'image': torch.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            'audio': torch.arange(
                audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            'label': None,
        }
        
        # forward pass
        with torch.no_grad():
          outputs = model(inputs=inputs, subsampled_output_points=subsampling)

        output = {k:v.cpu() for k,v in outputs.logits.items()}
        
        reconstruction['label'] = output['label']
        if 'image' not in reconstruction:
          reconstruction['image'] = output['image']
          reconstruction['audio'] = output['audio']
        else:
          reconstruction['image'] = torch.cat(
              [reconstruction['image'], output['image']], dim=1)
          reconstruction['audio'] = torch.cat(
              [reconstruction['audio'], output['audio']], dim=1)
          
        del outputs
        
  # finally, reshape image and audio modalities back to original shape
  reconstruction['image'] = torch.reshape(reconstruction['image'], images.shape)
  reconstruction['audio'] = torch.reshape(reconstruction['audio'], audio.shape)
  return reconstruction




# TODO : load some video


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", low_cpu_mem_usage=True)
model.to(device)

reconstruction = autoencode_video(video[None, :16], audio[None, :16*AUDIO_SAMPLES_PER_FRAME, 0:1])
