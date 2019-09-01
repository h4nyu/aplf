import typing as t


def r2_loss(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    length = gt.size(0)
    return ((((gt - pred)**2).sum() - ((gt - gt.mean())**2).sum())**2).log()/length



def r2(
    gt: t.Any,
    pred: t.Any,
) -> t.Any:
    return 1 - (((gt - pred)**2).sum())/(((gt - gt.mean())**2).sum())


