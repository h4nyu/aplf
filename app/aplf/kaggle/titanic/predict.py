from cytoolz.curried import keymap, filter, pipe, merge, map
import numpy as np
from torch.utils.data import DataLoader
import torch
from dask import delayed


def predict(model_path, dataset):
    model = torch.load(model_path)
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    outputs = []
    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        output = int(torch.argmax(model.forward(data)))
        outputs.append(output)

    return outputs


def evaluate(outputs, labels):
    return pipe(zip(outputs, labels),
                map(lambda x: x[0] == x[0]),
                list,
                np.array)
