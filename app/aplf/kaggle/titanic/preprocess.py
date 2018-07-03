from sklearn import preprocessing
from dask import delayed


@delayed
def label_encode(series, classes):
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    return le.transform(series)
