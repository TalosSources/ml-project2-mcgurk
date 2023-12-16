import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


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
    # each result is an array of 3 numpy arrays, each of size Ni,3  and Ni can be different

    n_plots = len(experiments)

    means = np.array([[np.mean(a, axis=0) for a in r] for r in results]) # both shape 3, 3, 3
    stds = np.array([[np.std(a, axis=0) for a in r] for r in results])

    # Creating the figure and the 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    width = 0.2
    spacing = 0.1
    colors = {'A' : 'blue', 'V' : 'yellow', 'A+V' : 'green'}
    labels = ['A', 'V', 'A+V']
    legend_labels = ['Audio', 'Visual', 'McGurk']

    # Plotting the bar charts with error bars
    for i in range(3):
        bars = []
        for j in range(3):
            bar =  axes[i].bar(np.arange(3) + (width+spacing)*j, means[i, :, j], yerr=stds[i, :, j], width=width, color=colors[labels[j]])
            bars.append(bar)

        axes[i].set_xticks(np.arange(3) + spacing+width)
        axes[i].set_xticklabels(labels)
        for ticklabel, tickcolor in zip(axes[i].get_xticklabels(), [colors[label] for label in labels]):
            ticklabel.set_color(tickcolor)

        axes[i].set_xlabel('Input video')
        axes[i].set_ylabel('Prediction averages')
        axes[i].legend(bars, legend_labels, title='Predicted phonemes', loc='upper right')
        axes[i].set_yscale('log')

    axes[0].set_title(f'Ba+Ga=Da Experiment')
    axes[1].set_title(f'Ba+Fa=Va Experiment')
    axes[2].set_title(f'Ga+Ba=Bga Experiment')
    
    # Adjusting layout
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()






# TODO: Plot the average confidence scores for each normal sample of each experiment -> If possible, with test set
# TODO: Plot also for McGurk samples
# also maybe TODO: plot the confidence increase from normal samples to mcgurk samples for the mcgurk syllable (if it's interesting) -> maybe on a logscale
# and #TODO at home with PC, test my shiny aggregate function, and the masked pipeline, and many steps and aggresive learning rate 