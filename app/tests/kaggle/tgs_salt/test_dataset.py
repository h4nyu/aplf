from aplf.kaggle.tgs_salt.dataset import TgsSaltDataset


def test_dataset():
    dataset = TgsSaltDataset(
        dataset_dir='/store/kaggle/tgs-salt',
    )
    assert len(dataset) == 4000
    x, y = dataset[0]
    print(y)


