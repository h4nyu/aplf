import typing as t


def r2(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    return 1 - ((pred - gt)**2).sum()/((pred - gt.mean())**2).sum()
