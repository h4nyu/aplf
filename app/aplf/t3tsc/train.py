from .data import read_table, Table, Dataset
from torch.utils.data import Subset, DataLoader
from mlboard_client.writers import Writer
from aplf.config import mlboard_url
import torch
import typing as t

Log = t.Dict[str, float]
def train_epoch(
    train_loader:DataLoader,
    val_loader:DataLoader,
) -> Log:
    device = torch.device('cuda')
    for hh_inputs, hv_inputs, masks in train_loader:
        hh_inputs, hv_inputs, masks = hh_inputs.to(device), hv_inputs.to(device), masks.to(device)
        print(hh_inputs.shape)
    return {'train_loss': 1, 'val_loss': 2}

def train_cv(
    table:Table,
    train_indices:t.Sequence[int],
    val_indices:t.Sequence[int],
    n_epochs: int,
) -> None:

    dataset = Dataset(table)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, pin_memory=True)
    val_loader = DataLoader(val_set, pin_memory=True)
    writer = Writer(
        mlboard_url,
        't3tsc'
    )

    for e in range(n_epochs):
        log = train_epoch(
            train_loader,
            val_loader,
        )
        #  writer.add_scalars(log)
