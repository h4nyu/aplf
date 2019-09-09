from torch.utils.data import DataLoader, Dataset, Subset
import multiprocessing
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
from ..data import load_model, save_model


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
    tr_dataset,
    ev_dataset,
    tr_indices,
    val_indices,
):
    if Path(path).is_file():
        model = load_model(path)
    else:
        model = Model(
            size_in=3805,
        )

    model = model.to(DEVICE)
    tr_set = Subset(tr_dataset, indices=tr_indices)
    val_set = Subset(tr_dataset, indices=val_indices)
    tr_all_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=1024,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1024,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

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
        num_workers=4,
    )
    tr_optim = Adam(
        model.parameters(),
    )

    dist_optim = Adam(
        model.parameters(),
        amsgrad=True,
    )
    best_score = 0.
    tr_reg_loss = 0.
    tr_dist_loss = 0.
    for i in range(10000):
        tr_loss, tr_r2_loss = train_epoch(
            model,
            tr_loader,
            tr_optim,
        )
        tr_dist_loss, = train_regulation(
            model,
            tr_all_loader,
            dist_optim,
        )

        ev_dist_loss, = train_regulation(
            model,
            ev_loader,
            dist_optim,
        )

        val_loss, = eval_epoch(
            model,
            val_loader,
        )
        if best_score < val_loss:
            best_score = val_loss
            save_model(model, path)

        logger.info(f"tr: {tr_loss}, {tr_r2_loss} dist:{tr_dist_loss + ev_dist_loss}, val: {val_loss} bs:{best_score}")




def train_epoch(
    model,
    loader,
    optimizer,
) -> t.Tuple[float, float]:
    batch_len = len(loader)
    preds = []
    labels = []
    sources = []
    sum_mse_loss = 0.
    for source, label in loader:
        source = source.to(DEVICE)
        label = label.to(DEVICE)
        sources.append(source)
        labels.append(label)
        pred = model(source).view(-1)
        loss = mse_loss(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_mse_loss += loss.item()

    labels = cat(labels).view(-1)
    sources = cat(sources)
    preds = model(sources).view(-1)
    return (sum_mse_loss/batch_len, r2(preds, labels).item())

def regular_loss(pred, lower, heigher):
    lower_mask = (pred < lower).to(pred.device)
    heigher_mask = (pred > heigher).to(pred.device)
    mask = (lower_mask | heigher_mask).float()
    masked_pred = pred * mask
    ans = lower * ones(*pred.size()).to(pred.device) * lower_mask.float()
    + heigher * ones(*pred.size()).to(pred.device) * heigher_mask.float()
    return mse_loss(masked_pred, ans)

def train_regulation(
    model,
    loader,
    optimizer,
) -> t.Tuple[float]:
    batch_len = len(loader)
    sum_loss = 0.
    for source, _ in loader:
        source = source.to(DEVICE)
        y = model(source).view(-1)
        loss = 0.1*regular_loss(y, -1, 5.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )


def train_distorsion(
    model,
    loader,
    optimizer,
) -> t.Tuple[float]:
    batch_len = len(loader)
    sum_loss = 0.
    for source, _ in loader:
        source = source.to(DEVICE)
        y = model(source).view(-1)
        y_mean = y.mean()
        y_std = y.std()
        loss = 0.1*(mse_loss(y_mean, tensor([2.02]).to(DEVICE)) + mse_loss(y_std, tensor([0.92]).to(DEVICE)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )

def train_pi(
    model,
    loader,
    optimizer,
) -> t.Tuple[float]:
    model.train()
    batch_len = len(loader)
    sum_loss = 0.
    for source, _ in loader:
        source0 = source.to(DEVICE)
        source1 = source.to(DEVICE)
        loss = mse_loss(
            model(source0).view(-1),
            model(source1).view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )


def eval_epoch(
    model: Model,
    loader,
) -> t.Tuple[float]:
    model.eval()
    batch_len = len(loader)
    sum_loss = 0
    preds = []
    labels = []
    sources = []
    with no_grad():
        for source, label in loader:
            sources.append(source)
            labels.append(label)

    sources = cat(sources).to(DEVICE)
    labels = cat(labels).to(DEVICE)
    preds = model(sources).view(-1)
    loss = r2(preds, labels)
    return (loss.item(), )


def pred(
    model: Model,
    dataset: Dataset,
) -> t.Tuple[float]:
    model.eval()
    model = model.eval().to(DEVICE)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1024,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
    )
    batch_len = len(loader)
    sum_loss = 0
    preds = []
    sources = []
    labels = []
    with no_grad():
        for source, label in loader:
            sources.append(source)
            labels.append(label)

    labels = cat(labels).to(DEVICE)
    sources = cat(sources).to(DEVICE)
    preds = model(sources).view(-1)
    loss = r2(preds, labels)
    logger.info(f"loss:{loss.item()}")
    return preds.detach().cpu().numpy()
