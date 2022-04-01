# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import math

import numpy as np
import torch
from tqdm import tqdm


def principal_component_analysis(data: np.ndarray, dimension: int = 2) -> torch.Tensor:
    tensor_data = torch.from_numpy(data)
    tensor_data = tensor_data - torch.mean(tensor_data, 0).expand_as(tensor_data)
    # u, sigma, vt = torch.svd(torch.t(tensor_data))  #  u, s, v
    u, _, _ = torch.svd(torch.t(tensor_data))

    pca_out = u[:, :dimension]

    return pca_out


def calculate_sigma(sigma_x: float, sigma_y: float, radians: float) -> np.ndarray:
    sigma_xy = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    sigma_radians = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(sigma_radians, np.dot(sigma_xy, sigma_radians.T))

    return sigma


def isotropic_gaussian_kernel(kernel_size: int, sigma: float, is_tensor: bool = False) -> np.ndarray:
    kernel_range_size = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    gaussian_x_vector, gaussian_y_vector = np.meshgrid(kernel_range_size, kernel_range_size)
    kernel = np.exp(-(gaussian_x_vector ** 2 + gaussian_y_vector ** 2) / (2. * sigma ** 2))

    if is_tensor:
        gaussian_kernel = torch.FloatTensor(kernel / np.sum(kernel))
    else:
        gaussian_kernel = kernel / np.sum(kernel)

    return gaussian_kernel


def anisotropic_gaussian_kernel(kernel_size: int, sigma_matrix: np.array, is_tensor: bool = False) -> np.ndarray:
    kernel_range_size = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    gaussian_x_vector, gaussian_y_vector = np.meshgrid(kernel_range_size, kernel_range_size)
    gaussian_x_vector = gaussian_x_vector.reshape((kernel_size * kernel_size, 1))
    gaussian_y_vector = gaussian_y_vector.reshape((kernel_size * kernel_size, 1))
    gaussian_xy_vector = np.hstack((gaussian_x_vector, gaussian_y_vector)).reshape(kernel_size, kernel_size, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(gaussian_xy_vector, inverse_sigma) * gaussian_xy_vector, 2))

    if is_tensor:
        gaussian_kernel = torch.FloatTensor(kernel / np.sum(kernel))
    else:
        gaussian_kernel = kernel / np.sum(kernel)

    return gaussian_kernel


def get_batch_kernel(batch_size: int, kernel_size: int, min_sigma: float, max_sigma: float, iso_prob: float, scaling: int, is_tensor: bool = False) -> np.ndarray:
    batch_kernel = np.zeros((batch_size, kernel_size, kernel_size))

    progress_bar = tqdm(total=batch_size, unit="kernel", desc="Generate gaussian kernel")
    for i in range(batch_size):
        if np.random.random() < iso_prob:
            sigma = np.random.random() * (max_sigma - min_sigma) + min_sigma
            batch_kernel[i] = isotropic_gaussian_kernel(kernel_size, sigma, is_tensor=is_tensor)
        else:
            radians = np.random.random() * math.pi * 2 - math.pi
            sigma_x = np.random.random() * (max_sigma - min_sigma) + min_sigma
            sigma_y = np.clip(np.random.random() * scaling * sigma_x, min_sigma, max_sigma)
            sigma_matrix = calculate_sigma(sigma_x, sigma_y, radians)
            batch_kernel[i] = anisotropic_gaussian_kernel(kernel_size, sigma_matrix, is_tensor=is_tensor)
        progress_bar.update(1)

    if is_tensor:
        return torch.FloatTensor(batch_kernel)
    else:
        return batch_kernel


def main(args) -> None:
    # Generate a large number of Gaussian convolution kernels
    batch_kernel = get_batch_kernel(args.batch_size, args.kernel_size, args.min_sigma, args.max_sigma, args.iso_prob, args.scaling, is_tensor=False)
    # Calculate how many rows of data there are
    batch_size = np.size(batch_kernel, 0)
    # Expand horizontally
    batch_kernel = batch_kernel.reshape((batch_size, -1))
    # Calculate PCA(principal component analysis) matrix
    pca_matrix = principal_component_analysis(batch_kernel, args.dimension).float()
    # Save PCA(principal component analysis) matrix
    torch.save(pca_matrix, args.pca_matrix_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly generate PCA matrix on dataset.")
    parser.add_argument("--batch_size", type=int, help="How many random Gaussian convolution kernels are included in the PCA matrix.")
    parser.add_argument("--kernel_size", type=int, help="Gaussian convolution kernel scale.")
    parser.add_argument("--min_sigma", type=float, help="Minimum variance in Gaussian processing.")
    parser.add_argument("--max_sigma", type=float, help="Maximum variance in Gaussian processing.")
    parser.add_argument("--iso_prob", type=float, help="Same-sex or opposite-sex Gaussian kernel probability.")
    parser.add_argument("--scaling", type=int, help="Anisotropic Gaussian Kernel Transform Coefficients.")
    parser.add_argument("--dimension", type=int, help="Principal component analysis dimension.")
    parser.add_argument("--pca_matrix_path", type=str, help="The final generated PCA matrix save location.")

    args = parser.parse_args()

    main(args)
