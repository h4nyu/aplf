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
no random_erase
+ center mask loss

