from aplf.kaggle.tgs_salt.dataset import TgsSaltDataset, load_dataset_df


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs-salt')
    assert len(dataset_df) == 4000
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000
    print(dataset[0])
