# Getting started

## Training
1. Train SparseOcc on SemanticKITTI with 8 GPUs
```bash
bash tools/dist_train.sh ./projects/configs/sparseocc/sparseocc_kitti.py 8
```

2. Train SparseOcc on OpenOccupancy
```bash
bash tools/dist_train.sh ./projects/configs/sparseocc/sparseocc_nusc_256.py 8
```


During the training process, the model is evaluated on the validation set after every epoch. The checkpoint with best performance will be saved. The output logs and checkpoints will be available at work_dirs/$CONFIG or at location specified in config.

## Evaluation
Evaluate with 8 GPUs:
```bash
bash tools/dist_test.sh $PATH_TO_CFG $PATH_TO_CKPT 8
```

