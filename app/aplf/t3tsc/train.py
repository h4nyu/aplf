from .data import read_table, Table, Dataset, get_batch_iou_binary as _get_batch_iou
from .models import Res34Unet
from .losses import lovasz_hinge as get_loss
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image, make_grid
from pathlib import Path
from uuid import uuid4
import torch.nn.functional as F
import torch.nn as nn
import torch
import typing as t
from functools import partial
from mlboard_client.writers import Writer

device = torch.device('cuda')
get_batch_iou = partial(_get_batch_iou, thresold=0.5)



def validate_epoch(
    loader:DataLoader,
    model:Res34Unet,
):
    model.eval()
    x_loss = nn.BCELoss()
    running_loss = 0.0
    running_iou = 0.0
    for x_images, y_images in loader:
        x_images, y_images = x_images.to(device), y_images.to(device)
        with torch.set_grad_enabled(False):
            logits = model(x_images)
            loss = get_loss(logits.squeeze(1), y_images.squeeze(1))
        pred_images = F.sigmoid(logits)
        running_loss += loss.item()
        running_iou += get_batch_iou(pred_images > 0.5, y_images)
        save_image(
            torch.cat(
                (
                    x_images[:,1,:, :].unsqueeze(1),
                    x_images[:,0,:, :].unsqueeze(1),
                    pred_images,
                    y_images.unsqueeze(1),
                ),
                0
            ),
            '/store/tmp/val.png'
        )

    return {
        'val_loss': running_loss / len(loader) ,
        'val_iou': running_iou / len(loader) ,
    }


Log = t.Dict[str, float]
def train_epoch(
    loader:DataLoader,
    model:Res34Unet,
    optimizer:t.Any,
) -> Log:
    model.train()
    x_loss = nn.BCELoss()
    running_loss = 0.0
    running_iou = 0.0
    for x_images, y_images in loader:
        x_images, y_images = x_images.to(device), y_images.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            logits = model(x_images)
            loss = get_loss(logits.squeeze(1), y_images.squeeze(1))
            loss.backward()
            optimizer.step()
        pred_images = F.sigmoid(logits)
        running_loss += loss.item()
        running_iou += get_batch_iou(pred_images > 0.5, y_images)


    return {
        'train_loss': running_loss / len(loader),
        'train_iou': running_iou / len(loader),
    }

def save(model:t.Any, path:Path) -> None:
    best_param = model.state_dict()
    torch.save(best_param, path)

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
    model_path:Path,
) -> None:

    dataset = Dataset(table)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=16, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
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

    best_iou = 0
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
        current_iou = val_log['val_iou']
        if  current_iou > best_iou:
            best_iou = current_iou
            save(model, model_path)
