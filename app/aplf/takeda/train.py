from torch.utils.data import DataLoader, Dataset
import multiprocessing
from torch.optim import Adam
from torch import device, no_grad
from torch.nn.functional import mse_loss
import typing as t
from logging import getLogger

from .models import Model
from .eval import r2, r2_loss


logger = getLogger("takeda.train")

def train_epoch(
    model:Model,
    dataset:Dataset,
    batch_size:int,
) -> t.Tuple[float]:
    cuda = device('cuda')
    model = model.train().to(cuda)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    batch_len = len(loader)

    optimizer = Adam(
        model.parameters(),
        amsgrad=True,
    )

    sum_loss = 0
    for source, ans in loader:
        source = source.to(cuda)
        ans = ans.to(cuda)
        y = model(source)
        loss = r2_loss(y.view(-1), ans.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    mean_loss = sum_loss / batch_len
    return (mean_loss, )

def eval_epoch(
    model:Model,
    dataset:Dataset,
    batch_size:int,
) -> t.Tuple[float]:
    cuda = device('cuda')
    model = model.train().to(cuda)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    batch_len = len(loader)
    sum_loss = 0
    with no_grad():
        for source, ans in loader:
            source = source.to(cuda)
            ans = ans.to(cuda)
            y = model(source)
            loss = r2(y.view(-1), ans.view(-1))
            sum_loss += loss.item()
    mean_loss = sum_loss / batch_len
    return (mean_loss, )
