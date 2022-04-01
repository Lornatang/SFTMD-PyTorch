import os

# Randomly generate PCA matrix on dataset
os.system("python ./create_pca_matrix.py --batch_size 30000 --kernel_size 21 --min_sigma 0.2 --max_sigma 4.0 --iso_prob 1.0 --scaling 3 --dimension 10 --pca_matrix_path ../results/pca_matrix.pth")

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DF2K/original/train --output_dir ../data/DF2K/SFTMD/train --image_size 480 --step 240 --num_workers 16")
os.system("python ./prepare_dataset.py --images_dir ../data/DF2K/original/valid --output_dir ../data/DF2K/SFTMD/valid --image_size 256 --step 256 --num_workers 16")
