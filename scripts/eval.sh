#!/usr/bin/env bash

gpus=1

data_name=DSIFN  #DSIFN
net_G=crossfusion_Transformer_1enc_1dec_64head  
split=test
project_name=CD_crossfusion_Transformer_1enc_1dec_64head_DSIFN_b8_lr0.01_train_val_200_linear_sgd
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


