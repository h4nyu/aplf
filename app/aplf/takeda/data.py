from pathlib import Path
import vaex
from vaex.dataframe import DataFrameArrays


def read_csv(path: Path) -> DataFrameArrays:
    return vaex.read_csv(path)
