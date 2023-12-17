# # Testing the McGurk effect on Machine Learning models
# 
# ### Defining the McGurk effect experiments

from experiments import McGurkExperiment

# Instantiate the list of experiments
experiments = [
    McGurkExperiment("ba", "ga", "da"), # ba (auditory) + ga (visual) = da  (fusioned sound)
    McGurkExperiment("ba", "fa", "va"), # ba (auditory) + fa (visual) = va  (fusioned sound)
    McGurkExperiment("ga", "ba", "bga") # ga (auditory) + ba (visual) = bga (combined sound)
]
masked_experiment = False

# ### Ensuring reproducible experiment results

# Set the seeds for the experiments to ensure reproducible results
import torch
torch.manual_seed(42)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


# ## Testing the effect on pretrained PerceiverIO models with regression mapping

# ### Training the models

from models import McGurkPerceiver

perceiver_models = []
for experiment in experiments:
    # Instantiate a Perceiver model for the given experiment
    model = McGurkPerceiver(experiment)
    perceiver_models.append(model)

for model in perceiver_models:
    print(model.name())
    # Train the models
    _, _, _, _ = model.train(epochs=40000, learning_rate=0.001, train_with_masks=masked_experiment)

# ### Testing the model on normal samples

testing_results = [] # (3, 2, N_i, 3)
for model in perceiver_models:
    print(model.name())
    A_predictions, V_predictions = model.test(test_with_masks=False)
    testing_results.append([A_predictions.detach().cpu().numpy(), V_predictions.detach().cpu().numpy()])

#print(testing_results)

# ### Generating the McGurk predictions

mcgurk_predictions = [] # (3, Ni, 3) -> could be made into (3, 1, Ni, 3)

# Test the models on McGurk effect videos
for model in perceiver_models:
    print(model.name())
    predictions = model.test_mcgurk()
    # Average the predictions\n",
    predictions = predictions.cpu().detach().numpy()
    #print(predictions)
    mcgurk_predictions.append(predictions)


# ### Results

import plot

# combine results
results = [] # should end up (3, 3, Ni, 3)
for i in range(len(experiments)):
    combined_result = [testing_results[i][0], testing_results[i][1], mcgurk_predictions[i]] # 3 Ni 3
    results.append(combined_result)

plot.plot_perceiver_experiment(experiments, results, path=f"plots/{'plot_masked_dummy' if masked_experiment else 'plot'}.jpg")


