from aplf.takeda.app import run, submit


if __name__ == '__main__':
    run(
        n_splits=10,
        fold_idx=1,
    )
    #  submit('/store/model-1')
