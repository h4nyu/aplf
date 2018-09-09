from aplf.kaggle.tgs.train import train
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split


def test_train():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    train_df, val_df = train_test_split(dataset_df)
    output_dir = '/store/tmp'
    train(
        model_id=1,
        model_path=f"{output_dir}/model.pt",
        train_dataset=TgsSaltDataset(train_df),
        val_dataset=TgsSaltDataset(val_df),
        epochs=200,
        batch_size=32,
        patience=10,
        base_size=10
    )
