from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    assert len(dataset_df) == 4000
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000
    assert len(dataset[0]) == 4
