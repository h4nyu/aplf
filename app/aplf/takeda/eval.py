import typing as t


def r2(
    pred: t.Any,
    gt: t.Any,
) -> t.Any:
    _gt = gt.view(-1)
    _pred = pred.view(-1)
    return 1 - (((_gt - _pred)**2)).sum()/(((_gt - _gt.mean())**2)).sum()
