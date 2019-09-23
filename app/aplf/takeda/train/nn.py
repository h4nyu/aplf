from torch.utils.data import DataLoader, Dataset, Subset
import multiprocessing
import torch
from sklearn.metrics import r2_score
from torch.optim import Adam
from torch import device, no_grad, randn, tensor, ones, cat
from torch.nn.functional import mse_loss, l1_loss
import numpy as np
from pathlib import Path
import typing as t
from logging import getLogger
from aplf.utils.decorators import skip_if

from ..models import Model
from ..eval import r2
from ..data import load_model, save_model, interpolate, TakedaDataset, flat_distorsion


logger = getLogger("takeda.train")

DEVICE = device('cuda')


def add_noise(means, device):
    def _inner(x):
        pos_mask = (randn(*x.size()) > 0.9).float().to(device)
        neg_mask = (pos_mask < 0.5).float()
        _means = tensor(means).float().to(device)
        return x * neg_mask + pos_mask * _means
    return _inner

def train(
    path:str,
    tr_set:TakedaDataset,
    val_set:TakedaDataset,
):
    if Path(path).is_file():
        model = load_model(path)
    else:
        model = Model(
            size_in=len(tr_set.x_columns),
        )

    model = model.double().to(DEVICE)

    tr_loader = DataLoader(
        dataset=tr_set,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1024,
        shuffle=False,
        pin_memory=True,
        num_workers=3,
    )
    tr_optim = Adam(
        model.parameters(),
    )

    best_score = 0.
    tr_reg_loss = 0.
    tr_dist_loss = 0.
    epoch = 0
    while True:
        tr_metrics = train_epoch(
            model,
            tr_loader,
            val_loader,
            tr_optim,
            epoch,
        )

        val_metrics = eval_epoch(
            model,
            val_loader,
        )
        if best_score < val_metrics[0]:
            best_score = val_metrics[0]
            save_model(model, path)

        epoch += 1
        logger.info(f"{epoch}: tr: {tr_metrics} val: {val_metrics} bs:{best_score}")




def train_epoch(
    model,
    main_loader,
    sub_loader,
    optimizer,
    epoch,
) -> t.Tuple[float, float]:
    model.train()
    count = 0
    preds:t.List[t.Any] = []
    labels:t.List[t.Any] = []
    sources:t.List[t.Any] = []
    sum_m_loss = 0.
    sum_s_loss = 0.
    for source, label in main_loader:
        source = source.to(DEVICE)
        label = label.to(DEVICE)

        sources.append(source)
        labels.append(label)

        optimizer.zero_grad()
        out = model(source).view(-1)
        loss = mse_loss(out, label)
        loss.backward()
        optimizer.step()
        sum_m_loss += loss.item()
        count += 1

    _labels = cat(labels).view(-1)
    _sources = cat(sources)
    _preds = model(_sources).view(-1)
    return (sum_m_loss/count, r2(_preds, _labels).item())

def regular_loss(pred, lower, heigher):
    lower_mask = (pred < lower).to(pred.device)
    heigher_mask = (pred > heigher).to(pred.device)
    mask = (lower_mask | heigher_mask).float()
    masked_pred = pred * mask
    ans = lower * ones(*pred.size()).to(pred.device) * lower_mask.float()
    + heigher * ones(*pred.size()).to(pred.device) * heigher_mask.float()
    return mse_loss(masked_pred, ans)


def eval_epoch(
    model: Model,
    loader,
) -> t.Tuple[float]:
    #  model.eval()
    batch_len = len(loader)
    sum_loss = 0
    preds:t.List[t.Any] = []
    labels:t.List[t.Any] = []
    sources:t.List[t.Any] = []
    with no_grad():
        for source, label in loader:
            sources.append(source)
            labels.append(label)

    _sources = cat(sources).to(DEVICE)
    _labels = cat(labels).to(DEVICE)
    _preds = model(_sources).view(-1)
    loss = r2(_preds, _labels)
    return (loss.item(), )


def pred(
    model: Model,
    dataset: Dataset,
) -> t.Tuple[float]:
    model.eval()
    loader = DataLoader(
        dataset=dataset,
        batch_size=1024,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
    )
    batch_len = len(loader)
    sum_loss = 0
    preds:t.List[t.Any] = []
    sources:t.List[t.Any] = []
    labels:t.List[t.Any] = []
    with no_grad():
        for source, label in loader:
            sources.append(source)
            labels.append(label)

    _labels = cat(labels).to(DEVICE)
    _sources = cat(sources).to(DEVICE)
    _preds = model(_sources).view(-1)
    loss = r2(preds, _labels)
    logger.info(f"loss:{loss.item()}")
    return _preds.detach().cpu().numpy()
