import typing as t


def r2(
    pred: t.Any,
    gt: t.Any,
) -> t.Any:
    return 1 - (((gt - pred)**2)).sum()/(((gt - gt.mean())**2)).sum()
