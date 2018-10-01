# 0
## condition
UNet feature 24 depth 3
val_split:0.2
lr 0.01
cyclic 5 epoch 0.5 - 1.0
batch_size 32
SGD
cross entropy
data clean

## result
54 epoch (1h 7m)
val iou 0.68
train iou 0.88
val loss 0.28
train loss 0.022

## memo
overfitted

# 1
## condition
RUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch: 0.5 - 1.0
SGD
cross entropy
data clean
augmentation: hflip

## result
400 epoch(4h 41m)
### peak at epoch 313
val iou 0.7897
train iou 0.883
val loss 0.18
train loss 0.021

### end at epoch 399
val iou 0.763
train iou 0.893
val loss 0.19
train loss 0.018


# 2
## condition
DUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.5 - 1.0
SGD
cross entropy
data clean

## result
val iou |0.77|0.74|
train iou |0.87|0.86|
val loss |0.027|0.025|
train loss |0.13|0.14|

# memo
デーコーダはパラメータが多いとoverfitを抑制できる

# 3
## condition
RUNet feature 8 depth 3
val_split:0.2
batch_size: 64
lr: 0.01
cyclic: 5 epoch 0.5 - 1.0
SGD
lovasz
data clean
augmentation: hflip

## result

