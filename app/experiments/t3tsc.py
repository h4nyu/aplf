from aplf.t3tsc.pipelines import main

if __name__ == '__main__':
    main(
        n_splits=4,
        n_epochs=1000,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
    )
