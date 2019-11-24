import typing as t
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

class Dataset(_Dataset):
    def __init__(self, table:Table) -> None:
        self.table = table
        self.rows = list(table.values())
        self.images:t.Dict[Path, t.Any] = {}

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
        hh_img, hv_img, an_img = self.__get_image(hh), self.__get_image(hv), self.__get_image(an)
        img = (np.array([hh_img, hv_img]) / 255).astype(np.float32)
        return (
            img,
            an_img
        )


IndexPair = t.Tuple[t.Sequence[int], t.Sequence[int]]
def kfold(table:Table, n_splits:int) -> t.Sequence[IndexPair]:
    kf = KFold(n_splits=n_splits)
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
