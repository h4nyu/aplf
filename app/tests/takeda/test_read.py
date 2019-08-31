from aplf.takeda.data import read_csv


def test_read() -> None:
    df = read_csv('/store/takeda/train.csv')
    print(type(df))
