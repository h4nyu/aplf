from cytoolz.curried import keymap, filter, pipe, merge


def map_kwargs(bind={}):
    def bind_kwargs(func):
        def wapper(*args):
            _dict = pipe(args,
                         filter(lambda x: isinstance(x, dict)),
                         merge,
                         keymap(lambda x: bind[x] if x in bind else x))
            _args = pipe(args,
                         filter(lambda x: not isinstance(x, dict)),
                         list)
            return func(*_args, **_dict)
        wapper.__name__ = func.__name__
        return wapper
    return bind_kwargs
