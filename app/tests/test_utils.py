import os
from aplf.utils import skip_if_exists
from pathlib import Path

@skip_if_exists('path')
def mock_func(path):
    return Path('/tmp/from_test.txt')

def test_skip_if_exists():
    path = Path('/tmp/mock.txt')
    path.touch()
    assert path.exists()
    new_path = mock_func(path=path)
    assert not Path('/tmp/from_test.txt').exists()
    path.unlink()
