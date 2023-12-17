import torch
import os
from tqdm import tqdm
from experiments import McGurkExperiment
from .utils import load_video_and_audio, autoencode_video, AUDIO_SAMPLES_PER_FRAME
from .log_reg import LogisticRegression
import numpy as np


def obtain_latents_a_v_av(videos_paths, device):
    from .perceiver import PerceiverForMultimodalAutoencoding

    perceiver_model = PerceiverForMultimodalAutoencoding.from_pretrained(
        "deepmind/multimodal-perceiver", low_cpu_mem_usage=False
    )
    perceiver_model.to(device)

    perceiver_model.perceiver.input_preprocessor.mask_probs = {
        "image": 1.0,
        "audio": 0.0,
        "label": 1.0,
    }
    X_a = obtain_mc_gurk_last_latents(
        videos_paths, device, perceiver_model=perceiver_model
    )

    perceiver_model.perceiver.input_preprocessor.mask_probs = {
        "image": 0.0,
        "audio": 1.0,
        "label": 1.0,
    }
    X_v = obtain_mc_gurk_last_latents(
        videos_paths, device, perceiver_model=perceiver_model
    )

    perceiver_model.perceiver.input_preprocessor.mask_probs = {
        "image": 0.0,
        "audio": 0.0,
        "label": 1.0,
    }
    X_av = obtain_mc_gurk_last_latents(
        videos_paths, device, perceiver_model=perceiver_model
    )

    X_a_v_av = torch.cat((X_a, X_v, X_av))
    #assert X_a_v_av.size() == (3 * len(videos_paths), 512)

    return X_a_v_av


def obtain_mc_gurk_last_latents(videos_paths, device, perceiver_model=None):
    if perceiver_model is None:
        from transformers import PerceiverForMultimodalAutoencoding

        perceiver_model = PerceiverForMultimodalAutoencoding.from_pretrained(
            "deepmind/multimodal-perceiver", low_cpu_mem_usage=False
        )
        perceiver_model.to(device)

    # big last_latents array X that will contain 128 video_paths.length latents, one for each video
    # last_latent should be of shape (latent_size) = 512
    latents = []

    for video_path in tqdm(
        videos_paths,
        desc="Inferring last hidden states of Perceiver latents from samples",
        leave=False,
    ):
        video, audio = load_video_and_audio(video_path=video_path)

        frames_taken = 16  # TODO : investigate this
        last_hidden_state = autoencode_video(
            video[None, :frames_taken],
            audio[None, : (frames_taken * AUDIO_SAMPLES_PER_FRAME), 0:1],
            perceiver_model,
            device,
            SAMPLES_PER_PATCH=frames_taken,
            output_reconstruction=False,
        )

        latents.append(last_hidden_state)

    return torch.stack(latents)


def training_pipeline(
    labels,
    X_cache_path=None,
    videos_paths=None,
    epochs=10,
    learning_rate=0.0001,
    X_save_path=None,
    device=None,
    model_save_path=None,
    train_with_masks=True,
):
    """
    Trains a logistic regression model on the last hidden states of latents of a pre-trained Perceiver model that infers input video data
    """
    assert videos_paths is not None or X_cache_path is not None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if X_cache_path is not None:
        # If no videos are provided, load the tensor of last hidden states of latents from cache
        X = torch.load(X_cache_path).to(device)
    else:
        # Else, load the videos and obtain the tensor of last hidden states of latents
        X = (
            obtain_latents_a_v_av(videos_paths, device).to(device)
            if train_with_masks
            else obtain_mc_gurk_last_latents(videos_paths, device).to(device)
        )

        # Save the tensor of last hidden states of latents
        if X_save_path is not None:
            torch.save(X, X_save_path)

    N = X.size(0)  # Number of samples
    D = X.size(1)  # Latent size from Perceiver

    # Load labels
    Y = torch.nn.functional.one_hot(torch.tensor(labels, dtype=int)).float()
    Y = Y.to(device)
    if train_with_masks:
        Y = torch.cat((Y,Y,Y))

    # Train a logistic regression model on the last hidden states of latents
    C = int(labels.max()) + 1
    classification_model = LogisticRegression(D, C).to(device)
    classification_model.train_log_reg(X, Y, epochs=epochs, learning_rate=learning_rate)

    # Save the trained model
    if model_save_path is not None:
        torch.save(classification_model, model_save_path)

    # Compute the accuracy of the trained model
    predictions = classification_model.predict(X)  # size = (N, 3)

    predicted_labels = torch.argmax(predictions, dim=1)
    Y_labels = torch.argmax(Y, dim=1)

    correct = predicted_labels == Y_labels
    #print("Wrongly classified videos:")
    #for i in range(len(correct)):
    #    if correct[i] == False:
    #        print(f"{videos_paths[i]} : {predictions[i]}")
    accuracy = correct.sum() / N

    print(f"Training accuracy: {accuracy * 100}%")

    return classification_model, X, Y, accuracy


class McGurkPerceiver:
    def __init__(self, experiment: McGurkExperiment):
        self.experiment = experiment

    def name(self):
        return f"Perceiver model for experiment {self.experiment.auditory} (auditory syllable) + {self.experiment.visual} (visual syllable) = {self.experiment.mcgurk} (expected McGurk syllable)"

    def train(self, epochs=100000, learning_rate=0.00002, train_with_masks=False):
        """
        Trains a logistic regression model on the last hidden states of latents of a pre-trained Perceiver model
        """

        videos_paths, labels = self.experiment.training_videos()

        latents_path = (
            f"cache/tensors/perceiver_latents_{self.experiment.to_str()}.pt"
            if not train_with_masks
            else f"cache/tensors/perceiver_latents_{self.experiment.to_str()}_masked.pt"
        )
        model_path = (
            f"cache/models/perceiver_log_reg_{self.experiment.to_str()}.pt"
            if not train_with_masks
            else f"cache/models/perceiver_log_reg_{self.experiment.to_str()}_masked.pt"
        )

        if os.path.exists(latents_path):
            # If the latents are already computed, load them from cache
            self.model, X, Y, accuracy = training_pipeline(
                labels=labels,
                videos_paths=videos_paths,
                X_cache_path=latents_path,
                epochs=epochs,
                learning_rate=learning_rate,
                model_save_path=model_path,
                train_with_masks=train_with_masks,
            )
            return self.model, X, Y, accuracy

        else:
            self.model, X, Y, accuracy = training_pipeline(
                labels=labels,
                videos_paths=videos_paths,
                X_save_path=latents_path,
                epochs=epochs,
                learning_rate=learning_rate,
                model_save_path=model_path,
                train_with_masks=train_with_masks,
            )
            return self.model, X, Y, accuracy

    def test_mcgurk(self):
        videos_paths = self.experiment.mcgurk_videos()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        latents_path = f"cache/tensors/perceiver_latents_mcgurk_{self.experiment.to_str()}.pt"

        if os.path.exists(latents_path):
            X = torch.load(latents_path)
        else:
            X = obtain_mc_gurk_last_latents(videos_paths=videos_paths, device=device).to(device)
            torch.save(X, latents_path)

        return self.model.predict(X)

    def test(self, test_with_masks=False):
        """
        return: If we test with mask, we return 3 floats : a, v, a+v correct percentages
        Else, we return 1 float, representing correct percentage.
        """
        videos_paths, labels = self.experiment.testing_videos()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        latents_path = (
            f"cache/tensors/perceiver_latents_{self.experiment.to_str()}_test.pt"
            if not test_with_masks
            else f"cache/tensors/perceiver_latents_{self.experiment.to_str()}_masked_test.pt"
        )

        if os.path.exists(latents_path):
            X = torch.load(latents_path)
        else: 
            X = (
                obtain_latents_a_v_av(videos_paths=videos_paths, device=device)
                if test_with_masks
                else obtain_mc_gurk_last_latents(videos_paths=videos_paths, device=device)
            )
            torch.save(X, latents_path)

        # labels
        labels = torch.tensor(labels, dtype=int).to(device)
        N = labels.size(0)
        if test_with_masks:
            labels = torch.cat((labels, labels, labels))

        # predictions
        predictions = self.model.predict(X)
        print(f"videos_paths = {videos_paths}")
        print(f"labels = {labels}")
        print(f"predictions : {predictions}")

        correct = predictions.argmax(dim=1) == labels
        if test_with_masks:
            accuracy_A = correct[:N].sum() / N
            accuracy_V = correct[N:2*N].sum() / N
            accuracy_AV = correct[2*N:3*N].sum() / N
            print(f"test accuracy A = {accuracy_A}")
            print(f"test accuracy V = {accuracy_V}")
            print(f"test accuracy AV = {accuracy_AV}")
        else:
            accuracy = correct / N
            print(f"test accuracy : {accuracy}")

        N = predictions.size(0)
        A_predictions = predictions[labels==0]
        V_predictions = predictions[labels==1]

        return A_predictions, V_predictions



    def clear_cache(self):
        """
        Clears the cache of the latents and the trained model
        """

        os.system(f"rm cache/tensors/perceiver_latents_{self.experiment.to_str()}*")
        os.system(f"rm cache/models/perceiver_log_reg_{self.experiment.to_str()}*")
