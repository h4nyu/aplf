import typing as t


def skip_if(func: t.Callable) -> t.Any:
    def wrapper(*args, **kwargs):
        print('--start--')
        func(*args, **kwargs)
        print('--end--')
    return wrapper
