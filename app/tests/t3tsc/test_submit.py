from aplf.t3tsc.submit import make_json
from pathlib import Path
from aplf.utils import Timer
import json


def test_make_json() -> None:
    json_data = make_json(
        Path('/store/t3tsc/train_annotations'),
        categories = {'ice':1}
    )
    with open(Path('/store/t3tsc/sample_submit.json'), "r") as f:
        expected_json_data = json.load(f)
    print(json_data['train_13']['ice'][0])
    print(expected_json_data['test_00']['ice'][0])

