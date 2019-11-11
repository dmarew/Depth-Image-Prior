import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())

from utils.inpainting_utils import *

# Load the left and right images
left_img_pil = load("data/input/left/tsukuba_daylight_L_00001.png")
right_img_pil = load("data/input/right/tsukuba_daylight_R_00001.png")

# ignore the last dimension I don't know what is wrong with it
left_img_np = pil_to_np(left_img_pil)[0:3, :, :]
left_img_np = np.moveaxis(left_img_np, 0, -1)
right_img_np = pil_to_np(right_img_pil)[0:3, :, :]
right_img_np = np.moveaxis(right_img_np, 0, -1)

# Load the disparity image from left to right
# disparity_img_pil = load("data/ground_truth/disparity/left/tsukuba_disparity_L_00001.png")
disparity_img_pil = load("data/ground_truth/disparity/right/tsukuba_disparity_R_00001.png")

disparity_img_np = (pil_to_np(disparity_img_pil)*255).squeeze().astype(int)

# show left and right images
fig = plt.figure()
columns = 3
rows = 2
fig.add_subplot(rows, columns, 1)
# plt.imshow(left_img_np[0, :, :].squeeze())
plt.imshow(disparity_img_np)
fig.add_subplot(rows, columns, 2)
# plt.imshow(left_img_np[1, :, :].squeeze())
plt.imshow(left_img_np)
fig.add_subplot(rows, columns, 3)
# plt.imshow(left_img_np[2, :, :].squeeze())
plt.imshow(right_img_np)


# warp left image to look like right
sudo_right = np.zeros(right_img_np.shape)
for row in range(right_img_np.shape[0]):
    for col in range(right_img_np.shape[1]):
        sudo_col = col - disparity_img_np[row, col]
        if sudo_col >= right_img_np.shape[1] or sudo_col < 0:
            continue
        sudo_right[row, sudo_col, :] = left_img_np[row, col, :]

# fig.add_subplot(rows, columns, 4)
# plt.imshow(sudo_right)
# plt.show()

import torch

# disp = torch.from_numpy(disparity_img_np)
# sample_grid = disp.unsqueeze(0)
# sample_grid = torch.meshgrid([torch.arange(0, 5), torch.arange(1, 10)])
# print(sample_grid.shape)

left_im = torch.from_numpy(np.moveaxis(left_img_np, 2, 0)).unsqueeze(0)
right_im = torch.from_numpy(np.moveaxis(right_img_np, 2, 0)).unsqueeze(0)
B, C, H, W = left_im.shape
row_space = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W).unsqueeze(0)
col_space = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1).unsqueeze(0)

print(row_space.shape, col_space.shape)
# print(row_space[:5, :5])
# print(col_space[:5, :5])

disp = torch.from_numpy(pil_to_np(disparity_img_pil))

col_space += disp*(255/W)*2

fig.add_subplot(rows, columns, 4)
plt.imshow(col_space.squeeze().numpy())

# create occlusion mask
occlusion_mask = torch.ones(left_im.shape)
for row in range(H):
    max_val = -2
    for col in range(W):
        if col_space[:, row, col] <= max_val:
            occlusion_mask[:, :, row, col] = 0
        else:
            max_val = col_space[:, row, col]

fig.add_subplot(rows, columns, 5)
plt.imshow(np.moveaxis(occlusion_mask.numpy().squeeze(), 0, -1))

grid = torch.zeros((1, H, W, 2))
grid[:, :, :, 1] = row_space
grid[:, :, :, 0] = col_space
sudo_right = torch.nn.functional.grid_sample(left_im, grid)

sudo_right = sudo_right*occlusion_mask

img = np.moveaxis(sudo_right.numpy().squeeze(), 0, -1)

fig.add_subplot(rows, columns, 6)
plt.imshow(img)
plt.show()





