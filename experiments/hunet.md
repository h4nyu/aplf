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
+ center mask loss

## result
### 190(2h 3m)
val iou 0.89
train iou 0.810
val loss 0.106
train loss 0.059
## memo
DUnetよりoverfitしにくい
center mask はあまり作用しない？

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
center mask loss
+ cetner maxpool

## result
