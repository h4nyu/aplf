# 6
## condition
DUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.5 - 1.0
SGD
lovasz
data clean

## result
val iou 0.90
train iou 0.79
val loss 0.10
train loss 0.027


# 7
## condition
DUNet feature 16 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.2 - 1.0
SGD
lovasz
data clean

## result
epoch 400
val iou 0.92
train iou 0.77
val loss 0.109
train loss 0.026


# 8
## object
## condition
DUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.2 - 1.0
SGD
lovasz
data clean
+ mean teacher
+ consistency_rampup: 50
+ consistency_loss: CrossEntropyLoss
+ consistency: 0.5
+ ema_decay: 0
+ consistency_input: [val_images]
+ consistency_noise: 


## result

