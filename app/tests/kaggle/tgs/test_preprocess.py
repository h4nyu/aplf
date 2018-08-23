from aplf import config
from aplf.kaggle.tgs.preprocess import rl_enc, cleanup
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from tensorboardX import SummaryWriter


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000


#  def test_iou():
#      dataset_df = load_dataset_df('/store/kaggle/tgs').dropna().head(5)
#      y_true = dataset_df['y_mask_true']
#      y_pred = dataset_df['y_mask_true']
#      assert mean_iou(y_pred, y_true) == 1


def test_cleanup():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset_df = cleanup(dataset_df)
    assert len(dataset_df) == 3808
