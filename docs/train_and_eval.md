# Getting started

## Training
```bash
bash tools/dist_train.sh $CONFIG 8
```
During the training process, the model is evaluated on the validation set after every epoch. The checkpoint with best performance will be saved. The output logs and checkpoints will be available at work_dirs/$CONFIG or at location specified in config.

## Evaluation
Evaluate with 1 GPU:
```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval=bbox
```
The single-GPU inference will print the current performance after each iteration, which can serve as a quick indicator.

Evaluate with 8 GPUs:
```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8
```

