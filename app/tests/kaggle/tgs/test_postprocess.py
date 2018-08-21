from aplf.kaggle.tgs.postprocess import dense_crf
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df


def test_dense_crf():
    dataset_df = load_dataset_df('/store/kaggle/tgs').dropna().head(5)
    dataset = TgsSaltDataset(dataset_df)
    sample = dataset[0]
    print(sample['image'].shape)
    aa = dense_crf(sample['image'].view(-1, 101, 101), sample['mask'].view(-1, 101, 101))
