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
        num_workers=3,
    )

    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1024,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    tr_loader = DataLoader(
        dataset=tr_set,
        batch_size=1024,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
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

    dist_optim = Adam(
        model.parameters(),
        amsgrad=True,
    )
    best_score = 0.
    tr_reg_loss = 0.
    tr_dist_loss = 0.
    for i in range(10000):
        tr_metrics = train_epoch(
            model,
            tr_loader,
            ev_loader,
            tr_optim,
        )

        val_metrics = eval_epoch(
            model,
            val_loader,
        )
        if best_score < val_metrics[0]:
            best_score = val_metrics[0]
            save_model(model, path)

        logger.info(f"tr: {tr_metrics} val: {val_metrics} bs:{best_score}")




def train_epoch(
    model,
    main_loader,
    sub_loader,
    optimizer,
) -> t.Tuple[float, float]:
    count = 0
    preds = []
    labels = []
    sources = []
    sum_m_loss = 0.
    sum_s_loss = 0.
    model = model.train()
    for (m_source, m_label),  (s_source, s_label) in zip(main_loader, sub_loader):
        m_source = m_source.to(DEVICE)
        m_label = m_label.to(DEVICE)
        s_source = s_source.to(DEVICE)
        s_label = s_label.to(DEVICE)

        sources.append(m_source)
        labels.append(m_label)

        m_out =model(m_source).view(-1)
        s_out =model(s_source).view(-1)
        m_loss = mse_loss(m_out, m_label)
        s_loss = regular_loss(s_out, -1, 5.) + regular_loss(m_out, -1., 5.)

        loss = m_loss + s_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_m_loss += m_loss.item()
        sum_s_loss += s_loss.item()
        count += 1

    labels = cat(labels).view(-1)
    sources = cat(sources)
    preds = model(sources).view(-1)
    return (sum_m_loss/count, sum_s_loss/count, r2(preds, labels).item())

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
