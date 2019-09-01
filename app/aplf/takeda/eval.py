import typing as t


def r2_loss(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    return (((gt - pred)**2).mean() - ((gt - gt.mean())**2).mean())**2



def r2(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    return 1 - (((gt - pred)**2).sum())/(((gt - gt.mean())**2).sum())


