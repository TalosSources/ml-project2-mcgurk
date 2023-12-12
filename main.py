import os
import pprint
from models import mcgurk_perceiver
import numpy as np

sp = 'dataset/train'
auditory = 'ba'
visual = 'ga'
mcgurk = 'da'

syllables = [auditory, visual, mcgurk]
persons = ['ismail', 'jad', 'olena']
p = len(persons)
n = 20

def obtain_n_first_videos(dir):
    return sorted([os.path.join(dir, file) for file in os.listdir(dir)])[:n]

videos_paths = []
for s in syllables:
    s_dir = os.path.join(sp, s)
    s_paths = [os.path.join(s_dir, video_path) for video_path in os.listdir(s_dir)] # all videos paths for syllable s
    for person in persons:
        s_person_paths = [video_path for video_path in s_paths if person in video_path]
        s_person_paths = sorted(s_person_paths)[:n]
        videos_paths += s_person_paths

pprint.pprint(videos_paths) # should be of size #syllable * #person * n = 9 * n

labels = np.array([0 for _ in range(p*n)] + [1 for _ in range(p*n)] + [2 for _ in range(p*n)])

pprint.pprint(labels)


classification_model, X, Y, accuracy = mcgurk_perceiver.training_pipeline(
    labels=labels,
    X_cache_path='cache/tensors/X_debug.pt',
    videos_paths=videos_paths,
    epochs=10000,
    learning_rate=0.0001,
    X_save_path=None,
    device=None,
    model_save_path=None,
    train_with_masks=False,
)

# debug path : 'cache/tensors/X_debug.pt'