from aplf.kaggle.tgs.predict import predict
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from aplf.kaggle.tgs.model import UNet
from sklearn.model_selection import train_test_split
import torch


def test_predict():
    dataset_df = load_dataset_df('/store/kaggle/tgs').sample(10)
    dataset = TgsSaltDataset(dataset_df)
    model = UNet()
    model_paths = ['/store/tmp/model.pt']
    torch.save(model, model_paths[0])

    predicted_df = predict(
        model_paths=model_paths,
        output_dir='/store/tmp',
        dataset=dataset
    )
    assert len(predicted_df) == len(dataset_df)
