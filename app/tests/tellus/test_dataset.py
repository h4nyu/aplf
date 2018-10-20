from aplf.tellus.dataset import load_dataset_df, get_row


def test_get_row():
    rows = get_row(
        base_path='/store/tellus/train',
        sat="LANDSAT",
        label_dir="positive",
        label=True
    )
    assert len(rows) == 1530


def test_dataset():
    df = load_dataset_df(dataset_dir='/store/tellus/train')
    assert len(df) == 247971
