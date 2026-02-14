#!/bin/bash

python plot_training_curves.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 100 \
    --adv_position end \
    --num_train_trajectories 8 \
    --output_path my_plot.png \