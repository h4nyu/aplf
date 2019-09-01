import typing as t


def r2(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    return 1 - (((gt - pred)**2).sum())/(((gt - gt.mean())**2).sum())


