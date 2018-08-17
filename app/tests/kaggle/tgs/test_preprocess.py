from aplf.kaggle.tgs.preprocess import rl_enc, mean_iou
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    idx = dataset_df.index.get_loc('b5e1371b3b')
    dataset = TgsSaltDataset(dataset_df)
    sample_id, depth, image, mask = dataset[idx]
    image_array = mask.view(101, 101).numpy()
    encoded = rl_enc(image_array)
    assert dataset_df['rle_mask'].iloc[idx] == encoded


def test_iou():
    dataset_df = load_dataset_df('/store/kaggle/tgs').dropna().head(5)
    y_true = dataset_df['y_mask_true']
    y_pred = dataset_df['y_mask_true']
    assert mean_iou(y_pred, y_true) == 1
