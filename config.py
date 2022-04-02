# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "RCAN_x2"
# PCA matrix address
pca_matrix_path = "results/pca_matrix.pth"
# Gaussian kernel size
kernel_size = 21
# Gaussian blur sigma value
sigma = 2.6
# Gaussian blur min sigma value
min_sigma = 0.2
# Gaussian blur max sigma value
max_sigma = 4.0


if mode == "train":
    # Dataset
    train_image_dir = f"data/DIV2K/RCAN/train"
    valid_image_dir = f"data/DIV2K/RCAN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 256
    batch_size = 32
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 515

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # CosineAnnealingWarmRestarts scheduler parameter
    lr_scheduler_T_0 = epochs // 4
    lr_scheduler_T_mult = 1
    lr_scheduler_eta_min = 1e-7

    print_frequency = 100

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/best.pth.tar"