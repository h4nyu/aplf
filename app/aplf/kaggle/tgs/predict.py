from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from skimage import io
import torch
from dask import delayed
import pandas as pd
from aplf import config
from .preprocess import rl_enc
from .dataset import TgsSaltDataset


def predict(model_paths,
            output_dir,
            dataset):

    writer = SummaryWriter(config["TENSORBORAD_LOG_DIR"])
    loader = DataLoader(dataset.train(), batch_size=1)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    models = pipe(model_paths,
                  map(torch.load),
                  map(lambda x: x.to(device)),
                  list)
    for m in models:
        m.eval()
    df = pd.DataFrame()

    sample_ids = []
    output_fns = []

    n_iter = 0
    for sample in loader:
        sample_id = sample['id'][0]
        image = sample['image'].to(device)

        output = pipe(models,
                      map(lambda x: x(image)),
                      reduce(lambda x, y: x + y),
                      lambda x: x / len(models))
        output = torch.argmax(output, dim=1).float()

        log_images = [image[0], output]
        if 'mask' in sample.keys():
            log_images.append(sample['mask'].to(device)[0])

        writer.add_image(
            f"{output_dir}/{sample_id}",
            vutils.make_grid(log_images, scale_each=True),
            n_iter
        )

        fn = f"{output_dir}/{sample_id}.png"
        torchvision.utils.save_image(
            output,
            fn
        )
        output_fns.append(fn)
        sample_ids.append(sample_id)
        n_iter += 1

    df['id'] = sample_ids
    df['y_mask_pred'] = output_fns
    df = df.set_index('id')
    return df
