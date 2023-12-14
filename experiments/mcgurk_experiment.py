from glob import glob
import numpy as np


class McGurkExperiment:
    """
    Class that represents a McGurk experiment.
    """

    def __init__(self, auditory, visual, mcgurk):
        """
        Creates a new McGurk experiment.

        Args:
            auditory (str): The auditory syllabe
            visual (str): The visual syllabe
            mcgurk (str): The expected McGurk syllabe
        """
        self.auditory = auditory
        self.visual = visual
        self.mcgurk = mcgurk

    def training_videos(self):
        """
        Returns a tuple of the form (`paths`, `labels`) where:
        - `paths` is a list of paths to training videos samples
        - `labels` is a list of labels for each video sample

        The labels are as follows:

        - `0`: Auditory syllabe
        - `1`: Visual syllabe
        - `2`: McGurk expected syllabe

        The paths are sorted by label.
        """
        paths = []
        # Fetch auditory syllabe samples
        auditory = glob(f"dataset/train/{self.auditory}/*.avi")
        paths += auditory
        visual = glob(f"dataset/train/{self.visual}/*.avi")
        paths += visual
        # Fetch expected McGurk syllabe samples
        mcgurk = glob(f"dataset/train/{self.mcgurk}/*.avi")
        paths += mcgurk

        # Fix paths for Windows that thinks it's special
        for i in range(len(paths)):
            paths[i] = str.replace(paths[i], '\\', '/')

        # Craft labels
        labels = np.concatenate(
            (np.zeros(len(auditory)), np.ones(len(visual)), 2.0 * np.ones(len(mcgurk)))
        )

        return (paths, labels)
    
    def test_videos(self):
        """
        Returns a tuple of the form (`paths`, `labels`) where:
        - `paths` is a list of paths to test videos samples
        - `labels` is a list of labels for each video sample

        The labels are as follows:

        - `0`: Auditory syllabe
        - `1`: Visual syllabe
        - `2`: McGurk expected syllabe

        The paths are sorted by label.
        """
        paths = []
        # Fetch auditory syllabe samples
        auditory = glob(f"dataset/test/{self.auditory}/*.avi")
        paths += auditory
        visual = glob(f"dataset/test/{self.visual}/*.avi")
        paths += visual
        # Fetch expected McGurk syllabe samples
        mcgurk = glob(f"dataset/test/{self.mcgurk}/*.avi")
        paths += mcgurk

        # Fix paths for Windows that thinks it's special
        for i in range(len(paths)):
            paths[i] = str.replace(paths[i], '\\', '/')

        # Craft labels
        labels = np.concatenate(
            (np.zeros(len(auditory)), np.ones(len(visual)), 2.0 * np.ones(len(mcgurk)))
        )

        return (paths, labels)

    def mcgurk_videos(self):
        """
        Returns a list of paths to McGurk videos samples.
        """
        return glob(f"dataset/mcgurk/{self.auditory}_{self.visual}_{self.mcgurk}/*.avi")

    def to_str(self):
        """
        Returns a string representation of the experiment.
        """
        return f"{self.auditory}_{self.visual}_{self.mcgurk}"
