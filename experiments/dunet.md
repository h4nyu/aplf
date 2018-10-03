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
epoch 250
val iou 0.92
train iou 0.77
val loss 0.109
train loss 0.026

# 8
## condition
DUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.2 - 1.0 -> no cyclic
SGD -> Adam
lovasz
data clean
scse -> cse


## result
epoch 799
val iou 0.902
train iou 0.754
val loss 0.110
train loss 0.026
## memo
7番目とあまり変わらない結果、
パラメータ数が少なさすぎる可能性あり





# 9
## condition
DUNet feature 8 depth 3 -> DUNet feature 16 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam
lovasz
cse -> scse
data clean


## result
epoch 799
val iou 0.901
train iou 0.813
val loss 0.100
train loss 0.024
## memo
パラメータ数が多いほどスコアが高い
adamがsgd+cyclicよりスコアがよい


# 9
## condition
DUNet feature 16 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam
lovasz
scse
data clean
augmentation: hflip -> hflip, random_erase
