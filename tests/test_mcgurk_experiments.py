from experiments import McGurkExperiment


def test_mcgurk_experiment_training_videos():
    """
    Tests the `McGurkExperiment.training_videos` method.
    """
    experiment = McGurkExperiment("ba", "ga", "da")
    paths, labels = experiment.training_videos()

    assert len(paths) == 30
    assert len(labels) == 30

    assert paths[0] == "dataset/train/ba/ba01.avi"
    assert labels[0] == 0.0

    assert paths[9] == "dataset/train/ga/ga10.avi"
    assert labels[9] == 1.0

    assert paths[19] == "dataset/train/da/da10.avi"
    assert labels[19] == 2.0

    assert paths[29] == "dataset/train/da/da20.avi"
    assert labels[29] == 2.0
