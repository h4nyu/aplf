from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from aplf.kaggle.tgs.train import train, boost_fit
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split

#
#  def test_train():
#      dataset_df = load_dataset_df('/store/kaggle/tgs')
#      train_df, val_df = train_test_split(dataset_df)
#      output_dir = '/store/tmp'
#      train(
#          model_id=1,
#          model_path=f"{output_dir}/model.pt",
#          train_dataset=TgsSaltDataset(train_df),
#          val_dataset=TgsSaltDataset(val_df),
#          epochs=1,
#          batch_size=1,
#          patience=10,
#      )


def test_boost_fit():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    train_df0, val_df0 = train_test_split(dataset_df)
    train_df1, val_df1 = train_test_split(dataset_df)

    train_datasets = pipe([train_df0, train_df1],
                          map(TgsSaltDataset),
                          list)

    val_datasets = pipe([val_df0, val_df1],
                        map(TgsSaltDataset),
                        list)

    output_dir = '/store/tmp'
    model_paths = []
    for i, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        model_path = boost_fit(
            prev_model_paths=model_paths,
            model_id=i,
            model_path=f"{output_dir}/model_{i}.pt",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=1,
            batch_size=3,
            patience=10,
            base_size=5,
        )
        model_paths.append(model_path)
    print(model_paths)
