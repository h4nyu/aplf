from .data import read_table, Table, Dataset, get_batch_iou as _get_batch_iou
from .models import Res34Unet
from .losses import lovasz_hinge
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch
import typing as t
from functools import partial
from mlboard_client.writers import Writer

device = torch.device('cuda')
get_batch_iou = partial(_get_batch_iou, classes=[0, 1])



def validate_epoch(
    loader:DataLoader,
    model:Res34Unet,
):
    model.eval()
    x_loss = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_score = 0.0
    for x_images, y_images in loader:
        x_images, y_images = x_images.to(device), y_images.to(device).long()
        with torch.set_grad_enabled(False):
            pred_images = model(x_images)
            loss = x_loss(pred_images, y_images)
        running_loss += loss.item()
        running_score += get_batch_iou(pred_images, y_images)

    return {
        'val_loss': running_loss / len(loader) ,
        'score': running_score / len(loader) ,
    }


Log = t.Dict[str, float]
def train_epoch(
    loader:DataLoader,
    model:Res34Unet,
    optimizer:t.Any,
) -> Log:
    model.train()
    x_loss = nn.CrossEntropyLoss()
    running_loss = 0.0
    for x_images, y_images in loader:
        x_images, y_images = x_images.to(device), y_images.to(device).long()

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred_images = model(x_images)
            loss = x_loss(pred_images, y_images)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()

    return {'train_loss': running_loss / len(loader) } 

def train_cv(
    table:Table,
    train_indices:t.Sequence[int],
    val_indices:t.Sequence[int],
    n_epochs: int,
    max_lr:float,
    min_lr:float,
    momentum:float,
    weight_decay: float,
    scheduler_step: int,
    writer:Writer,
) -> None:

    dataset = Dataset(table)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=16)
    val_loader = DataLoader(val_set, batch_size=1)
    model = Res34Unet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, scheduler_step,
        min_lr
    )

    for e in range(n_epochs):
        lr_scheduler.step(e)
        train_log = train_epoch(
            train_loader,
            model,
            optimizer,
        )
        val_log = validate_epoch(
            val_loader,
            model,
        )

        writer.add_scalars({**train_log, **val_log, "lr": lr_scheduler.get_lr()[0]}) #type: ignore
