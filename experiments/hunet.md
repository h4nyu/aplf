# 12
## condition
DUNet feature 16 depth 3 -> HUNet feature 32 depth 4
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam
lovasz
scse
data clean
augmentation: hflip 
no random_erase: reduce_rampup 0, erase_num 0

## result
### 187(2h 3m)
val iou 0.85
train iou 0.812
val loss 0.09
train loss 0.052

## memo
DUnetよりoverfitしにくい
DUnetよりはやく学習する

# 13
## condition
HUNet feature 64 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam
lovasz
scse
data clean
augmentation: hflip 
no random_erase: reduce_rampup 0, erase_num 0 -> random_erase: reduce_rampup 300, erase_num 5

## result
### 234(5h 22m)
val iou 0.792
train iou 0.84
val loss 0.103
train loss 0.065
## memo
パラメータが多い方がスコアが高い


# 14
## condition
HUNet feature 32 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam
lovasz
scse
data clean
augmentation: hflip 
random_erase: num 5, rampup 300

## result
### 190(2h 3m)
val iou 0.810
train iou 0.89
val loss 0.106
train loss 0.059
## memo
random_eraseはあった方よいが多すぎると悪影響



### 56649181-36f6-483d-94f1-eecb66cc8575
epoch: 347 
time: 8h14m
val_iou: 0.829
train_iou 0.907
val_loss 0.079
train loss 0.032
## memo
val_lossが更新

# 15
## condition
HUNet feature 32 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam lovasz
scse
data clean
augmentation: hflip 
random_erase: num 5, rampup 300, -> random_erase: remove rampup num 1
+ center mask loss

## result
### 0a9144d0-72b6-4556-aeec-f55d4b8ac427
epoch: 400
val_iou: 0.839
train_iou 0.925
val_loss 0.085
train loss 0.024

# 16
## condition
HUNet feature 32 depth 3  -> HUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam lovasz
scse
data clean
augmentation: hflip 
random_erase: num 1
center mask loss
mean teacher
consistency loss: mes
consistency input: val
ema_decay: 0.2
best teacher

## result
### 4dc02d82-3b20-413e-9112-c63144ae9bc9
epoch: 319
val_iou: 0.794
train_iou 0.856
val_loss 0.14
train loss 0.052
## note
consistencyにcross entropyやlovaszなど敏感なlossを使ってはいけない
mseなど滑らかなlossを使うべき
feature32が8よりスコアが良い



# 17
hunet-mean-teacher
## condition
HUNet feature 32 depth 3  -> HUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
no cyclic
Adam lovasz
scse
data clean
augmentation: hflip 
random_erase: num 1
center mask loss
mean teacher
consistency loss: mes
consistency input: val
+ consistency noise: val
ema_decay: 0.2
teacher update: batch loop

## result

# seg-set-0
hunet
## condition
HUNet feature 32 depht 3
scse
fold: 7
class_criterion: cross entorpy
data clean
augmentation: hflip 
random_erase: num 5
+ seg_loss
+ seg_loss_weight: 0.5

## result
val_iou: 0.839
train_iou: 0.941
train_loss: 0.018
train_loss: 0.0123
val_loss: 0.0863
## memo
outputがx = x + x*centerなのでcenterの影響を多く受けてしまう
conv2dで受ける


# seg-set-1
hunet
## condition
HUNet feature 32 depht 3
scse
fold: 7
optimizer: adam + amsgrad
class_criterion: cross entorpy
data clean
augmentation: hflip 
random_erase: num 5, p 1
seg_loss
seg_loss_weight: 0.5
+ consistency_loss: seg
+ consistency_input: val, no_label
+ consistency_loss_weight: 0.1
+ center_loss
+ center_loss_weight: 0.1
+ output: conv2d

## result
epoch:394(10h08m)
val_iou: 0.860
train_iou: 0.973
train_loss: 0.00633
val_loss: 0.0138
## memo
consistency_lossが効いている


# seg-set-2
hunet
## condition
HUNet feature 32 depht 3
scse
fold: 3
optimizer: adam + amsgrad
class_criterion: cross entorpy
data clean
augmentation: hflip 
random_erase: num 5, p 1
seg_loss
seg_loss_weight: 0.5
consistency_input: train, val, no_label
+ consistency_loss: center, seg
+ consistency_loss_weight: 0.2
center_loss
center_loss_weight: 0.2
output: conv2d

## result
epoch: 202(5h35m)
val_iou: 0.859
train_iou: 0.922
train_loss: 0.0074
val_loss: 0.013
## memo
consistency_lossのweightが多い方がoverfitしにくい
協会が端ギリギリのサンプルを見落としてる



# seg-set-3
hunet
## condition
HUNet feature 32 depht 3
scse
fold: 0
optimizer: adam + amsgrad
class_criterion: cross entorpy
seg_criterion: cross entorpy
data clean
augmentation: hflip 
random_erase: num 5, p 1
seg_loss
seg_loss_weight: 0.5
consistency_input: train, val, no_label
consistency_loss: center, seg
consistency_loss_weight: 0.2
consistency_criterion: mse
center_loss
center_loss_weight: 0.2
output: conv2d
+ padding: zeropad 10

## result


