from cytoolz.curried import keymap


def merge_kwargs(bind={}):
    def bind_kwargs(func):
        def wapper(*args):
            _dict = {}
            dicts = filter(lambda x: isinstance(x, dict), args)
            [_dict.update(d) for d in dicts]
            _dict = keymap(lambda x: bind[x] if x in bind else x)(_dict)
            _args = filter(lambda x: not isinstance(x, dict), args)
            return func(*_args, **_dict)
        wapper.__name__ = func.__name__
        return wapper
    return bind_kwargs
