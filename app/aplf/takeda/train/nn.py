from torch.utils.data import DataLoader, Dataset, Subset
import multiprocessing
from torch.optim import Adam
from torch import device, no_grad, randn, tensor, ones
from torch.nn.functional import mse_loss
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

@skip_if(
    lambda *args: Path(args[0]).is_file(),
    lambda *args: load_model(args[0]),
)
def train(
    path:str,
    tr_dataset,
    ev_dataset,
    tr_indices,
    val_indices,
):
    cuda = device('cuda')
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
    best_score = 0.
    tr_reg_loss = 0.
    for i in range(10000):
        tr_loss, tr_r2_loss = train_epoch(
            model,
            tr_loader,
        )
        val_loss, = eval_epoch(
            model,
            val_loader,
        )
        if val_loss > best_score:
            best_score = val_loss
            save_model(model, path)

        tr_reg_loss, = train_regulation(
            model,
            tr_all_loader,
        )

        ev_reg_loss, = train_regulation(
            model,
            ev_loader,
        )

        tr_dist_loss, = train_distorsion(
            model,
            tr_all_loader,
        )

        ev_dist_loss, = train_distorsion(
            model,
            ev_loader,
        )
        logger.info(f"tr: {tr_loss}, {tr_r2_loss} reg: {tr_reg_loss + ev_reg_loss} dist: {tr_dist_loss+ev_dist_loss} val: {val_loss} best:{best_score}")




def train_epoch(
    model,
    loader,
) -> t.Tuple[float, float]:
    cuda = device('cuda')
    batch_len = len(loader)
    optimizer = Adam(
        model.parameters(),
        amsgrad=True,
    )
    sum_loss = 0.
    sum_r2_loss = 0.
    for source, ans in loader:
        source = source.to(cuda)
        ans = ans.to(cuda)
        y = model(source)
        loss = mse_loss(y.view(-1), ans.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        r2_loss = r2(y.view(-1), ans.view(-1))
        sum_r2_loss += r2_loss.item()

    mean_loss = sum_loss / batch_len
    mean_r2_loss = sum_r2_loss / batch_len
    return (mean_loss, mean_r2_loss)

def regular_loss(pred, lower, heigher, mean=2.0248391529):
    lower_mask = (pred < lower).to(pred.device)
    heigher_mask = (pred > heigher).to(pred.device)
    mask = (lower_mask | heigher_mask).float()
    masked_pred = pred * mask
    ans = mean * ones(*pred.size()).to(pred.device) * lower_mask.float()
    + mean * ones(*pred.size()).to(pred.device) * heigher_mask.float()
    return mse_loss(masked_pred, ans)

def train_regulation(
    model,
    loader,
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
        y = model(source).view(-1)
        loss = regular_loss(y, -1, 5.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )


def train_distorsion(
    model,
    loader,
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
        y = model(source).view(-1)
        y_mean = y.mean()
        y_std = y.std()
        loss = mse_loss(y_mean, tensor([2.02]).to(cuda)) + mse_loss(y_std, tensor([0.92]).to(cuda))
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
    with no_grad():
        for source, ans in loader:
            source = source.to(cuda)
            ans = ans.to(cuda)
            y = model(source).view(-1)
            loss = r2(y, ans.view(-1))
            sum_loss += loss.item()
    mean_loss = sum_loss / batch_len
    return (mean_loss, )


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
