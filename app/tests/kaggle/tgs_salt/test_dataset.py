from aplf.kaggle.tgs_salt.dataset import TgsSaltDataset


def test_dataset():
    dataset = TgsSaltDataset(
        image_dir='/store/kaggle/tgs-salt/images',
        mark_dir='/store/kaggle/tgs-salt/masks'
    )
    assert len(dataset) == 4000
