import matplotlib.pyplot as plt
import numpy as np


def plot_perceiver_experiment(experiments, results, path=None):
    """
    * experiments is the classic experiment array
    * results is an array of 3 results, one for each experiment.
    * a result: 
        -control prediction array for auditory input
        -control prediction array for visual input
        -prediction array for mcgurk input.

    a prediction array is a numpy array, either of shape (N, 3) (or (3, N), equal)
    containing confidences for A, V, MG syllable that the input could be.
    or, it could be a shape (3,) array, but it would then need to come with a standard_deviation

    -> we do one subplot per experiment. each subplot contains therefore 3x3 bars.
    it's very similar to the av-hubert ones. we could re-use code and make sur to share visual information
    """
    results_bgd, results_bfv = results
    


def plot_mcgurk_confidences():
    # Plot the average confidence scores for each sound of each experiment
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Define the axises labels and plot title
    #ax.set_title('Average confidence scores per syllable for each experiment')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Average confidence score per syllable')

    # Define bars for each type of syllables
    ind = np.arange(len(experiments))  # the x locations for the groups
    width = 0.3
    plt.bar(ind - width, model_predictions_avg[:,0], width, label='Auditory syllable')
    plt.bar(ind, model_predictions_avg[:,1], width, label='Visual syllable')
    plt.bar(ind + width, model_predictions_avg[:,2], width, label='McGurk effect syllable')

    plt.xticks(ind + width / 2, (experiments[0].to_str(), experiments[1].to_str(), experiments[2].to_str()))
    plt.legend(bbox_to_anchor=(1.1, 0.5), loc='center left')
    plt.yscale('log')
    plt.show()

def plot_test_confidences():
    # certifies the model is potent at recognizing syllables
    ...

def plot_mgurk_confidence_increase():
    # maybe, only if it's information rich
    ...

plot_mcgurk_confidences()


# TODO: Plot the average confidence scores for each normal sample of each experiment -> If possible, with test set
# TODO: Plot also for McGurk samples
# also maybe TODO: plot the confidence increase from normal samples to mcgurk samples for the mcgurk syllable (if it's interesting) -> maybe on a logscale
# and #TODO at home with PC, test my shiny aggregate function, and the masked pipeline, and many steps and aggresive learning rate 