import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
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
left_disp_img_pil = load("data/ground_truth/disparity/left/tsukuba_disparity_L_00001.png")
right_disp_img_pil = load("data/ground_truth/disparity/right/tsukuba_disparity_R_00001.png")

left_disp_img_np = (pil_to_np(left_disp_img_pil)*255).squeeze().astype(int)
right_disp_img_np = (pil_to_np(right_disp_img_pil)*255).squeeze().astype(int)

# # show left and right images
fig = plt.figure()
columns = 4
rows = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(left_disp_img_np)
fig.add_subplot(rows, columns, 2)
plt.imshow(right_disp_img_np)
fig.add_subplot(rows, columns, 3)
plt.imshow(left_img_np)
fig.add_subplot(rows, columns, 4)
plt.imshow(right_img_np)

left_im = torch.from_numpy(np.moveaxis(left_img_np, 2, 0)).unsqueeze(0)
right_im = torch.from_numpy(np.moveaxis(right_img_np, 2, 0)).unsqueeze(0)
right_disp = torch.from_numpy(pil_to_np(right_disp_img_pil))
left_disp = torch.from_numpy(pil_to_np(left_disp_img_pil))

def warp(input_img, disp_img):
    B, C, H, W = input_img.shape
    row_space = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W).unsqueeze(0)
    col_space = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1).unsqueeze(0)

    col_space += disp_img * (255 / W) * 2

    # create occlusion mask
    occlusion_mask = torch.ones(left_im.shape)
    for row in range(H):
        max_val = -2
        for col in range(W):
            if col_space[:, row, col] <= max_val:
                occlusion_mask[:, :, row, col] = 0
            else:
                max_val = col_space[:, row, col]

    grid = torch.zeros((1, H, W, 2))
    grid[:, :, :, 1] = row_space
    grid[:, :, :, 0] = col_space
    output_img = torch.nn.functional.grid_sample(input_img, grid)

    return output_img, occlusion_mask

sudo_right, occlusion_mask = warp(left_im, right_disp)

sudo_right = sudo_right*occlusion_mask

img = np.moveaxis(sudo_right.numpy().squeeze(), 0, -1)

warped_depth1, _ = warp(left_disp.unsqueeze(0), right_disp)
warped_depth1 = warped_depth1.numpy().squeeze() - right_disp.numpy().squeeze()
warped_depth2, _ = warp(right_disp.unsqueeze(0), -left_disp)
warped_depth2 = warped_depth2.numpy().squeeze() - left_disp.numpy().squeeze()
# warped_depth2, _ = warp(right_im, -left_disp)
# warped_depth2 = np.moveaxis(warped_depth2.numpy().squeeze(), 0, -1)

fig.add_subplot(rows, columns, 5)
plt.imshow(np.moveaxis(occlusion_mask.numpy().squeeze(), 0, -1))

fig.add_subplot(rows, columns, 6)
plt.imshow(img)

fig.add_subplot(rows, columns, 7)
plt.imshow(warped_depth1)

fig.add_subplot(rows, columns, 8)
plt.imshow(warped_depth2)

plt.show()





