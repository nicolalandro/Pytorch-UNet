# UNet
How to train and predict it.

```
CUDA_VISIBLE_DEVICES="0,1" nohup python3.7 train.py --amp --batch-size 4 --epochs 50 > train_daedalus.log 2>&1 &
# into wandb folder there are outputs in checkoint there are models

python3.7 predict_daedalus.py --output ./results --model checkpoints/checkpoint_epoch1.pth
# in the results folder there are images

python3.7 test_deaedalus_fiftyone.py
```