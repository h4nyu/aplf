from .data import read_csv, TakedaDataset, kfold
from torch.utils.data import Subset
from .models import Model
from .train import train_epoch, eval_epoch
from logging import getLogger
logger = getLogger("takeda.app")

def run(n_splits:int, fold_idx:int)->None:
    df = read_csv('/store/takeda/train.csv')
    dataset = TakedaDataset(df)
    indices = kfold(dataset, n_splits=n_splits)
    train_set = Subset(dataset, indices[0][0])
    val_set = Subset(dataset, indices[0][1])

    model = Model(
        size_in=3805,
    )
    while True:
        train_loss, = train_epoch(
            model=model,
            dataset=train_set,
            batch_size=256
        )

        val_loss, = eval_epoch(
            model=model,
            dataset=val_set,
            batch_size=2048,
        )
        logger.info(f'train loss: {train_loss} val loss: {val_loss}')


def run_lgb():
    df = read_csv('/store/takeda/train.csv')
    dataset = TakedaDataset(df)
    indices = kfold(dataset, n_splits=n_splits)
    train_set = Subset(dataset, indices[0][0])
    val_set = Subset(dataset, indices[0][1])

