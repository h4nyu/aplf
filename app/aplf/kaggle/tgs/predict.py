from cytoolz.curried import keymap, filter, pipe, merge, map
import torchvision
from torch.utils.data import DataLoader
from skimage import io
import torch
from dask import delayed


def predict(model_path,
            output_dir,
            dataset):
    model = torch.load(model_path)
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    outputs = []

    for ids, depth, image, mask in loader:
        image = image.to(device)
        mask = mask.to(device)
        output = model(image)
        #  output[output > 0.5] = 1
        #  output[output <= 0.5] = 0
        grid = torchvision.utils.make_grid(
            [output[:, 0, :, :], mask[:, 0, :, :]],
        )
        torchvision.utils.save_image(
            grid,
            f"{output_dir}/{ids[0]}.png"
        )

    return outputs
