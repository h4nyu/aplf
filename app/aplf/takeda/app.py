from .data import read_csv, TakedaDataset
from .models import Model
from .train import train_epoch

def run()->None:
    df = read_csv('/store/takeda/train.csv')
    dataset = TakedaDataset(df)
    model = Model(
        size_in=3805,
    )
    while True:
        train_epoch(
            model=model,
            dataset=dataset,
            batch_size=1024,
        )


