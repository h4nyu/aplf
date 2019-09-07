import typing as t
from mypy_extensions import (Arg, DefaultArg, NamedArg,
                             DefaultNamedArg, VarArg, KwArg)


def skip_if(chek_skip:t.Callable[[VarArg(t.Any), KwArg(t.Any)], bool]) -> t.Any:
    def decorator(func):
        def wrapper(*args, **kwargs) -> t.Optional[t.Any]:
            if not chek_skip(*args, **kwargs):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator
