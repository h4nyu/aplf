from .data import read_csv, TakedaDataset, kfold, create_dataset, save_model, load_model, TakedaPredDataset, save_submit
from torch.utils.data import Subset
from .models import Model
from .train.lgbm import train
from logging import getLogger
import pickle
logger = getLogger("takeda.app")


def run(
    n_splits: int,
    fold_idx: int,
    lgbm_params,
) -> None:
    df = read_csv('/store/takeda/train.csv')
    indices = kfold(df, n_splits=n_splits)
    tr_set = create_dataset(df.iloc[indices[0][fold_idx]])
    val_set = create_dataset(df.iloc[indices[1][fold_idx]])
    train(
        tr_set,
        val_set,
        lgbm_params,
        path=f"/store/lgbm-model-{n_splits}-{fold_idx}.pkl"
    )


def submit(path:str) -> None:
    df = read_csv('/store/takeda/test.csv')
    with open(path, 'rb') as f:
        model = pickle.load(f)

    preds = model.predict(df)

    save_submit(
        df,
        preds,
        '/store/submit.csv'
    )
