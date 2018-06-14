
def dict_kwargs(func):
    def wapper(*args):
        _dict = {}
        dicts = filter(lambda x: isinstance(x, dict), args)
        [_dict.update(d) for d in dicts]
        _args = filter(lambda x: not isinstance(x, dict), args)
        return func(*_args, **_dict)
    return wapper
