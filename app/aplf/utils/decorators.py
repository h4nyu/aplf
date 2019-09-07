import typing as t
from mypy_extensions import (Arg, DefaultArg, NamedArg,
                             DefaultNamedArg, VarArg, KwArg)


def skip_if(
    check_skip:t.Callable[[VarArg(t.Any), KwArg(t.Any)], bool],
    skip_return:t.Callable[[VarArg(t.Any), KwArg(t.Any)], t.Any]=lambda *a, **kw:None,
) -> t.Any:
    def decorator(func):
        def wrapper(*args, **kwargs) -> t.Optional[t.Any]:
            if not check_skip(*args, **kwargs):
                result = func(*args, **kwargs)
                return result
            else:
                return skip_return(*args, **kwargs)
        return wrapper
    return decorator
