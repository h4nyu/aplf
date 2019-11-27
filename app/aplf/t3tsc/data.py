import typing as t
import Augmentor
from pathlib import Path
from glob import glob
import re
from skimage import io
import numpy as np

from skimage.transform import rescale, resize as _resize
from sklearn.model_selection import KFold
from torch.utils.data import Dataset as _Dataset
from multiprocessing import Pool, cpu_count
Row = t.Tuple[Path, Path, Path]
Table = t.Dict[str, Row]

def extract_id(fname:str) -> str:
    res = re.search(r"\d+",fname)
    if res is not None:
        return res.group()
    else:
        return ""

def read_table(
    x_dir:Path,
    y_dir:Path,

) -> Table:
    assert x_dir.is_dir()
    assert y_dir.is_dir()

    ids  = [
        extract_id(Path(x).name)
        for x
        in glob(f"{x_dir}/*")
    ]

    table = {
        x: (
            x_dir.joinpath(f"train_hh_{x}.jpg"),
            x_dir.joinpath(f"train_hv_{x}.jpg"),
            y_dir.joinpath(f"train_{x}.png"),
        )
        for x
        in ids
    }
    return table
def to_binary(mask:t.Any) -> t.Any:
    return ((mask >= 1) & (mask <= 10)).astype(np.float32)

class Dataset(_Dataset):
    def __init__(self, table:Table, mode="train") -> None:
        self.table = table
        self.rows = list(table.values())
        self.images:t.Dict[Path, t.Any] = {}
        self.mode = mode

    def __len__(self) -> int:
        return len(self.rows)

    def __get_image(self, path:Path) -> t.Any:
        if path in self.images:
            return self.images[path]
        else:
            image = io.imread(path)
            self.images[path] = image
            return image

    def __getitem__(self, idx:int) -> t.Tuple[t.Any, t.Any]:
        row = self.rows[idx]
        hh, hv, an = row
        hh_img, hv_img, mask = self.__get_image(hh), self.__get_image(hv), self.__get_image(an)
        if self.mode == 'train':
            hh_img, hv_img, mask = train_aug([hh_img, hv_img, mask], probability=0.5)

        img = (np.array([hh_img, hv_img]) / 255).astype(np.float32)
        mask = to_binary(mask).astype(np.float32)

        return (
            img, mask
        )


IndexPair = t.Tuple[t.Sequence[int], t.Sequence[int]]
def kfold(table:Table, n_splits:int) -> t.Sequence[IndexPair]:
    kf = KFold(n_splits=n_splits, random_state=0)
    indices = range(len(table))
    return kf.split(indices)


def resize(inpath:Path, outpath:Path, size:float) -> Path:
    if outpath.exists():
        return outpath
    image = io.imread(inpath)
    image = _resize(image, size)
    io.imsave(outpath, image)
    return outpath

def resize_all(
    in_dir:Path,
    out_dir:Path,
    size:t.Tuple[int, int],
    pattern:str="*.jpg"
) -> t.Sequence[Path]:
    out_dir.mkdir(exist_ok=True)
    args = [
        (Path(x), out_dir.joinpath(Path(x).name), size)
        for x
        in glob(f"{in_dir}/{pattern}")
    ]
    with Pool(cpu_count()) as p:
        return p.starmap(resize, args)


def get_iou(pred:t.Any, label:t.Any, classes=t.Sequence[int]) -> float:
    ious = []
    for i in classes:
        union = ((label == i) | (pred == i)).sum().item()
        if union == 0:
            ious.append(0)
        else:
            intersection = ((label == i) & (pred == i)).sum().item()
            ious.append(intersection / union)
    return np.array(ious).mean()


def get_batch_iou_binary(preds:t.Any, labels:t.Any, thresold:float=0.5) -> float:
    ious = [
        get_iou(p > thresold, l, classes=[0, 1])
        for p, l
        in zip(preds, labels)
    ]
    return np.array(ious).mean()

def get_batch_iou(preds:t.Any, labels:t.Any, classes=t.Sequence[int]) -> float:
    ious = [
        get_iou(p, l, classes)
        for p, l
        in zip(preds, labels)
    ]
    return np.array(ious).mean()

def horizontal_flip(image):
    return np.flip(image, axis=1).copy()

def train_aug(images, probability=0.5):
    p = Augmentor.DataPipeline([images])
    p.rotate90(probability=probability)
    p.rotate270(probability=probability)
    p.flip_left_right(probability=probability)
    p.flip_top_bottom(probability=probability)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    return p.sample(1)[0]
