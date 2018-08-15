from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from .predict import predict
from sklearn.model_selection import train_test_split
from dask import delayed


class Graph(object):
    def __init__(self,
                 dataset_dir,
                 output_dir,
                 epochs,
                 batch_size,
                 val_split_size,
                 patience,
                 ):
        dataset_df = delayed(load_dataset_df)(dataset_dir)
        splited = delayed(train_test_split)(
            dataset_df,
            test_size=val_split_size,
            shuffle=False,
        )
        train_dataset = delayed(TgsSaltDataset)(
            delayed(lambda x: x[0])(splited)
        )

        val_dataset = delayed(TgsSaltDataset)(
            delayed(lambda x: x[1])(splited)
        )
        trained = delayed(train)(
            model_path=f"{output_dir}/model.pt",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        eval_train = delayed(predict)(
            model_path=delayed(lambda x: x['model_path'])(trained),
            output_dir=output_dir,
            dataset=train_dataset
        )

        progress_file = delayed(lambda x: x['progress'].to_json(f"{output_dir}/progress.json"))(trained)

        self.output = delayed(lambda x: x)((
            progress_file,
            eval_train
        ))
