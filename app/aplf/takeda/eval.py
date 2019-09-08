import typing as t


def r2_loss(
    pred: t.Any,
    gt: t.Any,
) -> t.Any:
    return (((gt - pred)**2).mean() - ((gt - gt.mean())**2).mean()).abs()


def r2(
    pred: t.Any,
    gt: t.Any,
) -> t.Any:
    return 1 - (((gt - pred)**2).sum())/(((gt - gt.mean())**2).sum())
