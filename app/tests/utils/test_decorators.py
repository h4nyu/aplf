import pytest
from aplf.utils.decorators import skip_if


@pytest.mark.parametrize('is_skip, expected', [
    (True, None),
    (False, 0),
])
def test_skip_if(is_skip, expected) -> None:
    def func():
        return 0
    func = skip_if(lambda *a, **kw: is_skip)(func)
    assert func() == expected
