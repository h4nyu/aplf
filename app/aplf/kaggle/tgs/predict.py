from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
import torchvision
from torch.utils.data import DataLoader
from skimage import io
import torch
from dask import delayed
import pandas as pd
from .preprocess import rl_enc


def predict(model_paths,
            output_dir,
            dataset):

    loader = DataLoader(dataset, batch_size=1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    models = pipe(model_paths,
                  map(torch.load),
                  map(lambda x: x.to(device)),
                  list)
    for m in models:
        m.eval()
    df = pd.DataFrame(columns=['id', 'rle_mask'])

    sample_ids = []
    rle_masks = []

    for ids, depth, image in loader:
        image = image.to(device)
        output = pipe(models,
                      map(lambda x: x(image)),
                      reduce(lambda x, y: x + y),
                      lambda x: x / len(models))
        output = torch.argmax(output, dim=1).float()
        images = [image[:, 0, :, :], output, ]
        grid = torchvision.utils.make_grid(
            images
        )
        sample_ids.append(ids[0])
        rle_masks.append(rl_enc(output[0, :, :].cpu().numpy()))
        torchvision.utils.save_image(
            grid,
            f"{output_dir}/{ids[0]}.png"
        )
    df['id'] = sample_ids
    df['rle_mask'] = rle_masks

    return df
