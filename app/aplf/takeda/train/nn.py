from torch.utils.data import DataLoader, Dataset, Subset
import multiprocessing
from torch.optim import Adam
from torch import device, no_grad, randn, tensor, ones, cat
from torch.nn.functional import mse_loss, l1_loss
import numpy as np
from pathlib import Path
import typing as t
from logging import getLogger
from aplf.utils.decorators import skip_if

from ..models import Model
from ..eval import r2, r2_loss
from ..data import load_model, save_model


logger = getLogger("takeda.train")



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
    cuda = device('cuda')
    if Path(path).is_file():
        model = load_model(path)
    else:
        model = Model(
            size_in=3805,
        )

    model = model.to(cuda)
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
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    tr_loader = DataLoader(
        dataset=tr_set,
        batch_size=1024,
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
        amsgrad=True,
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
        val_loss, = eval_epoch(
            model,
            val_loader,
        )
        if val_loss > best_score:
            best_score = val_loss
            save_model(model, path)

        #  tr_dist_loss, = train_regulation(
        #      model,
        #      tr_all_loader,
        #      dist_optim,
        #  )
        #
        logger.info(f"tr: {tr_loss}, {tr_r2_loss} dist:{tr_dist_loss} val: {val_loss} best:{best_score}")




def train_epoch(
    model,
    loader,
    optimizer,
) -> t.Tuple[float, float]:
    cuda = device('cuda')
    batch_len = len(loader)
    preds = []
    labels = []
    for source, label in loader:
        source = source.to(cuda)
        label = label.to(cuda)
        pred = model(source)
        preds.append(pred)
        labels.append(label)
    preds = cat(preds)
    labels = cat(labels)
    loss = mse_loss(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (loss.item(), r2(preds, labels).item())

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
    cuda = device('cuda')
    batch_len = len(loader)
    sum_loss = 0.
    for source, _ in loader:
        source = source.to(cuda)
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
    cuda = device('cuda')
    batch_len = len(loader)
    sum_loss = 0.
    for source, _ in loader:
        source = source.to(cuda)
        y = model(source).view(-1)
        y_mean = y.mean()
        y_std = y.std()
        loss = 0.1*(mse_loss(y_mean, tensor([2.02]).to(cuda)) + mse_loss(y_std, tensor([0.92]).to(cuda)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )

def train_pi(
    model,
    loader,
    aug,
    weight,
) -> t.Tuple[float]:
    cuda = device('cuda')
    batch_len = len(loader)
    optimizer = Adam(
        model.parameters(),
        amsgrad=True,
    )
    sum_loss = 0.
    for source, _ in loader:
        source = source.to(cuda)
        source_0 = aug(source)
        source_1 = aug(source)
        loss = weight * mse_loss(
            model(source_1).view(-1),
            model(source_0).view(-1),
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
    cuda = device('cuda')
    model.eval()
    batch_len = len(loader)
    sum_loss = 0
    preds = []
    labels = []
    with no_grad():
        for source, label in loader:
            source = source.to(cuda)
            label = label.to(cuda)
            pred = model(source).view(-1)
            preds.append(pred)
            labels.append(label)

    preds = cat(preds)
    labels = cat(labels)
    loss = r2(preds, labels)
    return (loss.item(), )


def pred(
    model: Model,
    dataset: Dataset,
) -> t.Tuple[float]:
    model.eval()
    cuda = device('cuda')
    model = model.train().to(cuda)
    loader = DataLoader(
        dataset=dataset,
        batch_size=2048,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
    )
    batch_len = len(loader)
    sum_loss = 0
    preds = []
    with no_grad():
        for source, _ in loader:
            source = source.to(cuda)
            y = model(source).view(-1)
            preds += y.tolist()
    return np.array(preds)
