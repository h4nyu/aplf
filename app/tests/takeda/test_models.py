from aplf.takeda.models import Model
from torch import tensor, rand
import torchvision

def test_model() -> None:
    model = Model(
        size_in=10,
    )
    batch_size = 10
    x = rand(batch_size, 10)
    y = model(x)
    assert y.size() == (batch_size, 1)
