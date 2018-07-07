from torch.utils.data import DataLoader
import torch
from dask import delayed


@delayed
def predict(model, dataset):
    loader = DataLoader(dataset)
    device = torch.device("cuda")
    labels = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        output = model(data)
        labels.append(output.detach().numpy()[0][0])
    return labels
