from torch.utils.data import DataLoader, Dataset
import multiprocessing
from torch.optim import Adam
from torch import device
import typing as t
from logging import getLogger

from .models import Model
from .eval import r2


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
        num_workers=4,
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
        loss = (- r2(ans.view(-1), y.view(-1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    mean_loss = sum_loss / batch_len
    logger.info(f'mean_loss: {mean_loss}')
    return (mean_loss, )
