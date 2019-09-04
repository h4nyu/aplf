from .data import read_csv, TakedaDataset, kfold, create_dataset, save_model, load_model, TakedaPredDataset, save_submit
from torch.utils.data import Subset
from .models import Model
from .train.nn import train_epoch, eval_epoch, pred_epoch
from logging import getLogger
logger = getLogger("takeda.app")


def run(n_splits: int, fold_idx: int) -> None:
    df = read_csv('/store/takeda/train.csv')
    dataset = TakedaDataset(df)
    indices = kfold(dataset, n_splits=n_splits)
    train_set = Subset(dataset, indices[fold_idx][0])
    val_set = Subset(dataset, indices[fold_idx][1])

    model = Model(
        size_in=3805,
    )

    best_val_loss = 0
    while True:
        train_loss, train_r2_loss = train_epoch(
            model=model,
            dataset=train_set,
            batch_size=1024
        )

        val_loss, = eval_epoch(
            model=model,
            dataset=val_set,
            batch_size=2048,
        )
        logger.info(f'train loss: {train_r2_loss} val loss: {val_loss}')
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'/store/model-{fold_idx}')

def submit(model_path:str) -> None:
    df = read_csv('/store/takeda/test.csv')
    model = load_model(model_path)
    dataset = TakedaPredDataset(df)
    preds = pred_epoch(
        model,
        dataset,
        2048,
    )
    save_submit(
        df,
        preds,
        '/store/submit.csv'
    )

