from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from sklearn.model_selection import train_test_split
from dask import delayed


class Graph(object):
    def __init__(self,
                 dataset_dir,
                 model_path,
                 val_split_size=0.25,
                 epochs=10):

        dataset_df = delayed(load_dataset_df)(dataset_dir)
        splited = delayed(train_test_split)(
            dataset_df,
            test_size=val_split_size
        )
        train_dataset = delayed(TgsSaltDataset)(
            delayed(lambda x: x[0])(splited)
        )

        val_dataset = delayed(TgsSaltDataset)(
            delayed(lambda x: x[1])(splited)
        )
        trained = delayed(train)(
            model_path=model_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
        )
        progress_file = delayed(lambda df: df.to_json(f"{dataset_dir}/progress.json"))(trained)

        self.output = delayed(lambda x: x)((
            progress_file
        ))
