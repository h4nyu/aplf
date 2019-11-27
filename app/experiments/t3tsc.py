from aplf.t3tsc.pipelines import main
from datetime import datetime

if __name__ == '__main__':

    main(
        workspace_name=f't3tsc-{datetime.utcnow()}',
        n_splits=5,
        n_epochs=2000,
        max_lr=0.01,
        min_lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler_step=5,
        aug={
            'crop_random': {"percentage_area":0.8}
        }
    )
