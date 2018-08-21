from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
import torch.nn.functional as F
from sklearn.metrics import jaccard_similarity_score
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
            dataset,
            log_interval=100,
            ):

    writer = SummaryWriter(config["TENSORBORAD_LOG_DIR"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
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
    rle_masks = []
    scores = []

    n_itr = 0
    for sample in loader:
        sample_id = sample['id'][0]
        image = sample['image'].to(device)

        output = pipe(models,
                      map(lambda x: x(image)),
                      map(lambda x: F.softmax(x, dim=1)),
                      reduce(lambda x, y: x + y))
        output = torch.argmax(output, dim=1).float()
        sample_ids.append(sample_id)
        rle_masks.append(rl_enc(output.cpu().numpy().reshape(101, 101)))

        log_images = [image[0], output]
        if 'mask' in sample.keys():
            mask = sample['mask'].to(device)[0]
            log_images.append(mask)
            score = jaccard_similarity_score(
                output.cpu().numpy().reshape(-1), mask.cpu().numpy().reshape(-1))
            scores.append(score)

        if n_itr % log_interval == 0:
            writer.add_image(
                f"{output_dir}/{sample_id}",
                vutils.make_grid(log_images, scale_each=True),
            )

        n_itr += 1

    df['id'] = sample_ids
    df['rle_mask'] = rle_masks
    if len(scores) > 0:
        df['score'] = scores
    df = df.set_index('id')
    return df
