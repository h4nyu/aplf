from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
import torch.nn.functional as F
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
from .metric import iou


def predict(model_paths,
            dataset,
            log_dir,
            log_interval=100,
            ):

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

    with torch.no_grad():
        for sample in loader:
            sample_id = sample['id'][0]
            image = sample['image'].to(device)

            normal_outputs = pipe(
                models,
                map(lambda x: x(image)[0]),
                list,
            )
            fliped_outputs = pipe(
                models,
                map(lambda x: x(image.flip([3]))[0].flip([3])),
                list,
            )
            output = pipe(
                [*normal_outputs, *fliped_outputs],
                map(lambda x: x.softmax(dim=1)),
                reduce(lambda x, y: x + y / 2),
                lambda x: F.softmax(x, dim=1),
                lambda x: x.argmax(dim=1).float()
            )


            sample_ids.append(sample_id)
            rle_masks.append(rl_enc(output.cpu().numpy().reshape(101, 101)))

            log_images = [image[0], output]
            if 'mask' in sample.keys():
                mask = sample['mask'].to(device)[0]
                log_images.append(mask)
                score = iou(output.cpu().numpy(), mask.cpu().numpy())
                scores.append(score)

            if n_itr % log_interval == 0:
                with SummaryWriter(log_dir) as w:
                    w.add_image(
                        f"predict",
                        vutils.make_grid(log_images, scale_each=True),
                        n_itr
                    )

            n_itr += 1

        df['id'] = sample_ids
        df['rle_mask'] = rle_masks
        if len(scores) > 0:
            df['score'] = scores
            score = df['score'].mean()
            with SummaryWriter(log_dir) as w:
                w.add_text('score', f'score: {score}')
        df = df.set_index('id')
        return df
