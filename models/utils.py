import torch
import numpy as np

import os
import re

import ssl
import tempfile

from urllib import request

from tqdm import tqdm


import cv2
import imageio
import scipy.io.wavfile


# Utilities to fetch videos from UCF101 dataset
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

# CONSTANTS? TODO
AUDIO_SAMPLES_PER_FRAME = 48000 // 25


def list_ucf_videos():
    """Lists videos available in UCF101 dataset."""
    global _VIDEO_LIST
    if not _VIDEO_LIST:
        index = (
            request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
        )
        videos = re.findall("(v_[\w_]+\.avi)", index)
        _VIDEO_LIST = sorted(set(videos))
    return list(_VIDEO_LIST)


def fetch_ucf_video(video):
    """Fetchs a video and cache into local filesystem."""
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print("Fetching %s => %s" % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path


# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def load_video_and_audio(index=0, video_path=None):
    if video_path is None:
        video_names = list_ucf_videos()
        video_path = fetch_ucf_video(video_names[index])

    # Extract audio using FFMPEG and encode as pcm float wavfile (only format readable by scipy.io.wavfile).
    os.system(
        f"""ffmpeg -y -hide_banner -loglevel error -i "{video_path}"  -c copy  -f wav -map 0:a cache/tmp/pcm_f32le -ar 48000 cache/tmp/audio.wav"""
    )

    sample_rate, audio = scipy.io.wavfile.read("cache/tmp/audio.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 2**15
    elif audio.dtype != np.float32:
        raise ValueError(
            "Unexpected datatype. Model expects sound samples to lie in [-1, 1]"
        )

    video = load_video(video_path)

    return video, audio


def save_gif(images, path="./animation.gif"):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(path, converted_images, fps=25)


def save_audio(data, sample_rate=48000, path="./audio.wav"):
    scipy.io.wavfile.write(path, sample_rate, data)


def average_latents(latents):
    # expect a latent of shape (num_latents, latent_size)
    after = torch.mean(latents, dim=1).reshape((512))
    return after

def aggregate_latents(latents):
    # expect a latent of shape (784, 512). we know 784 = 112 * 7 = 56 * 14
    k = 14
    l = int(784 / k)
    latents = latents.reshape((784, 512))
    tensors = []
    for i in range(k):
        tensors.append(torch.mean(latents[i*l:(i+1)*l], dim=0).reshape((512)))
    return torch.cat(tensors).reshape((k * 512))


def autoencode_video(
    images, audio, model, device, SAMPLES_PER_PATCH=16, output_reconstruction=False
):
    # only create entire video once as inputs
    inputs = {
        "image": torch.from_numpy(np.moveaxis(images, -1, 2)).float().to(device),
        "audio": torch.from_numpy(audio).to(device),
        "label": torch.zeros((images.shape[0], 700)).to(device),
    }

    nchunks = 128
    reconstruction = {}
    for chunk_idx in tqdm(range(nchunks if output_reconstruction else 1), leave=False):
        image_chunk_size = np.prod(images.shape[1:-1]) // nchunks
        audio_chunk_size = audio.shape[1] // SAMPLES_PER_PATCH // nchunks
        subsampling = {
            "image": torch.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)
            ),
            "audio": torch.arange(
                audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)
            ),
            "label": None,
        }

        # forward pass
        with torch.no_grad():
            outputs = model(
                inputs=inputs,
                subsampled_output_points=subsampling,
                return_dict=True,
                output_hidden_states=True,
            )

        output = {k: v.cpu() for k, v in outputs.logits.items()}
        #last_hidden_state = average_latents(outputs.hidden_states[-1])
        last_hidden_state = average_latents(outputs.hidden_states[-1])

        if output_reconstruction:
            reconstruction["label"] = output["label"]
            if "image" not in reconstruction:
                reconstruction["image"] = output["image"]
                reconstruction["audio"] = output["audio"]
            else:
                reconstruction["image"] = torch.cat(
                    [reconstruction["image"], output["image"]], dim=1
                )
                reconstruction["audio"] = torch.cat(
                    [reconstruction["audio"], output["audio"]], dim=1
                )

        del outputs

    if output_reconstruction:
        # finally, reshape image and audio modalities back to original shape
        reconstruction["image"] = torch.reshape(reconstruction["image"], images.shape)
        reconstruction["audio"] = torch.reshape(reconstruction["audio"], audio.shape)
        return reconstruction
    else:
        return last_hidden_state
