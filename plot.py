import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

def mean_and_std(x, axis=0): # x.shape (Ni, 3)
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    return median, mad


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

    means_and_stds = np.array([[mean_and_std(a, axis=0) for a in r] for r in results]) # shape 3, 3, 2, 3

    # Creating the figure and the 3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), constrained_layout=False)

    fig.supxlabel('Input Video content')
    fig.supylabel('Mean regression confidence')

    width = 0.2
    spacing = 0.05
    colors = {'A' : 'blue', 'V' : 'orange', 'A+V' : 'green'}
    labels = ['A', 'V', 'A+V']
    legend_labels = ['Audio', 'Visual', 'McGurk']

    # Plotting the bar charts with error bars
    for i in range(3):
        bars = []
        for j in range(3):
            r = means_and_stds[i, :, :, j]
            mean = r[:, 0]
            std = r[:, 1]
            print(f"mean = {mean}, std = {std}")
            bar =  axes[i].bar(np.arange(3) + (width+spacing)*j, mean, yerr=std, width=width, color=colors[labels[j]])
            bars.append(bar)

        axes[i].set_xticks(np.arange(3) + spacing+width)
        axes[i].set_xticklabels(labels)
        for ticklabel, tickcolor in zip(axes[i].get_xticklabels(), [colors[label] for label in labels]):
            ticklabel.set_color(tickcolor)

        axes[i].set_yscale('log')

    axes[0].set_title(f'Ba+Ga=Da Experiment')
    axes[1].set_title(f'Ba+Fa=Va Experiment')
    axes[2].set_title(f'Ga+Ba=Bga Experiment')
    plt.legend(bars, legend_labels, bbox_to_anchor=(1.05, 1.0), title='Predicted phonemes')
    
    # Adjusting layout
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, format='jpg', transparent=False)
    else:
        plt.show()
        

def plot_human_control(results, path=None, dpi='600'):
    """
    Results is a numpy array of shape (5,), with ratios of samples working on us for
    ba_ga_da, ba_fa_va, ga_ba_bga, words_BGD and words_BFV experiments
    """
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,3), constrained_layout=True)

    plt.ylim([0, 1])

    width = 0.6
    spacing = 0.05

    # Plotting the bar charts with error bars
    bar =  ax.bar(np.arange(5), results, width=width, color='Blue')

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['Ba+Ga=Da', 'Ba+Fa=Va', 'Ga+Ba=Bga', 'B..+G..=D..', 'B..+F..=V..'], size='small')

    ax.set_xlabel('McGurk Experiment', size='medium', fontweight='bold')
    ax.set_ylabel('Ratio of apparent McGurk Effect', size='medium', fontweight='bold')

    # Adjusting layout
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, format='jpg', transparent=False, dpi=dpi)
    else:
        plt.show()
