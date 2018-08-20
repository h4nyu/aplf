from aplf.kaggle.tgs.predict import predict
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from aplf.kaggle.tgs.model import UNet, TinyUNet
from sklearn.model_selection import train_test_split
import torch


def test_predict():
    dataset_df = load_dataset_df('/store/kaggle/tgs').sample(10)
    dataset = TgsSaltDataset(dataset_df)
    model0 = UNet(in_ch=1)
    model1 = UNet(in_ch=3)
    model_paths = ['/store/tmp/model0.pt', '/store/tmp/model1.pt']
    torch.save(model0, model_paths[0])
    torch.save(model1, model_paths[1])

    predicted_df = predict(
        model_paths=model_paths,
        dataset=dataset,
    )
    print(predicted_df)
    assert len(predicted_df) == len(dataset_df)
