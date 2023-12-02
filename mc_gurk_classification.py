import torch
import tqdm
from utils import *
import os

class LogisticRegression(torch.nn.Module):
     
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
    def forward(self, x):
        outputs = torch.softmax(self.linear(x), dim=-1)
        return outputs
     
    """
    @params :
        X : tensor of shape (N, d) where N is the amount of mcgurk videos (or subsamples of which), and d is perceiver latent_size (=512)
        Y : tensor of shape (N), of integers in [C], where C is the number of possible syllables.
        model : LogisticRegression model
    @return: nothing
    """
    def train_log_reg(self, X, Y, epochs=200000, learning_rate=0.01):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for iter in tqdm(range(epochs), desc='Training Epochs'):

            optimizer.zero_grad()
            outputs = self(X)
            #if(outputs.size(1) == 1):
            #    outputs = outputs.reshape(outputs.size(0))
            loss = criterion(outputs, Y)

            loss.backward()

            optimizer.step()

            if(iter % 5000 == 0):
                print(f"iter {iter}")
                print(f"    loss={loss}")

        # nothing else to do if we want a minimal thing

def sorted_videos_paths(videos_dir):
    return [os.path.join(videos_dir, video_path) for video_path in sorted(os.listdir(videos_dir))]

def obtain_latents_a_v_av(videos_paths, device):

    from models import PerceiverForMultimodalAutoencoding
    perceiver_model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", low_cpu_mem_usage=False)
    perceiver_model.to(device)
    
    perceiver_model.perceiver.input_preprocessor.mask_probs = {"image": 1.0, "audio": 0.0, "label": 1.0}
    X_a = obtain_mc_gurk_last_latents(videos_paths, device, perceiver_model=perceiver_model)

    perceiver_model.perceiver.input_preprocessor.mask_probs = {"image": 0.0, "audio": 1.0, "label": 1.0}
    X_v = obtain_mc_gurk_last_latents(videos_paths, device, perceiver_model=perceiver_model)

    perceiver_model.perceiver.input_preprocessor.mask_probs = {"image": 0.0, "audio": 0.0, "label": 1.0}
    X_av = obtain_mc_gurk_last_latents(videos_paths, device, perceiver_model=perceiver_model)

    X_a_v_av = torch.cat((X_a, X_v, X_av))
    assert X_a_v_av.size() == (3 * len(videos_paths), 512)

    return X_a_v_av


def obtain_mc_gurk_last_latents(videos_paths, device, perceiver_model=None):

    if perceiver_model is None:
        from models import PerceiverForMultimodalAutoencoding
        perceiver_model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", low_cpu_mem_usage=False)
        perceiver_model.to(device)
    
    #big last_latents array X that will contain 128 video_paths.length latents, one for each video
    #last_latent should be of shape (latent_size) = 512
    latents = []

    for video_path in videos_paths:
        print(f"video {video_path}")
        video, audio = load_video_and_audio(video_path=video_path)

        frames_taken = 16 # TODO : investigate this
        last_hidden_state = autoencode_video(
            video[None, :frames_taken], 
            audio[None, :frames_taken*AUDIO_SAMPLES_PER_FRAME, 0:1], 
            perceiver_model, 
            device, 
            SAMPLES_PER_PATCH=frames_taken,
            output_reconstruction=False
        )

        latents.append(last_hidden_state)

    return torch.stack(latents)

def read_labels(label_path="labels.txt"):
    import csv
    labels = []
    with open(label_path, 'r') as lf:
        lines = lf.readlines()
    C = int(lines[0])
    for line in lines[1:]:
        labels.append(int(line))

    Y = torch.nn.functional.one_hot(torch.tensor(labels, dtype=int)).float()
    return C, Y
    
 

def training_pipeline(X_cache_path="X_cached.pt", 
                      labels_path='labels.txt', 
                      videos_paths=None, epochs=10, 
                      learning_rate=0.0001,
                      X_save_path=None, 
                      device=None, 
                      model_save_path=None):
    
    assert videos_paths is not None or X_cache_path is not None
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if videos_paths is not None:

        X = obtain_mc_gurk_last_latents(videos_paths, device).to(device)

        if X_save_path is not None:
            torch.save(X, X_save_path)
    else:
        X = torch.load(X_cache_path).to(device)

    N = X.size(0)
    D = X.size(1)

    #C = 2 # TODO : load class count from somewhere, surely label file also, or maybe something like numpy.distinct
    #Y = torch.randint(0, C, size=(N,), device=device) # TODO : load label file from dataset

    C, Y = read_labels(labels_path)
    Y = Y.to(device)
    print(f"got C = {C}, Y = {Y}")

    classification_model = LogisticRegression(D, C).to(device)
    classification_model.train_log_reg(X, Y, epochs=epochs, learning_rate=learning_rate)

    for i in range(N):
        x = X[i]
        y = Y[i]
        prediction = classification_model(x)
        print(f"prediction : {prediction}, ground_truth : {y}")

    if model_save_path is not None:
        torch.save(classification_model, model_save_path)

    return classification_model