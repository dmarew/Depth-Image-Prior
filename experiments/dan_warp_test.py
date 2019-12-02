import time
import torch.nn.functional
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image


# Load the left and right images
left_img_pil = Image.open("../data/input/left/tsukuba_daylight_L_00001.png")
right_img_pil = Image.open("../data/input/right/tsukuba_daylight_R_00001.png")
# Load the disparity image from left to right
disparity_img_pil = Image.open("../data/ground_truth/disparity/right/tsukuba_disparity_R_00001.png")

left_img  = transforms.functional.to_tensor(left_img_pil)[:3, :, :].unsqueeze(0)
disparity = transforms.functional.to_tensor(disparity_img_pil)

_, C, H, W = left_img.shape
row_space = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W).unsqueeze(0).unsqueeze(3)
col_space = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1).unsqueeze(0).unsqueeze(3)
col_space += disparity.view(1, H, W, 1) 
print(col_space.max(), col_space.min())
#row_space = disparity.view(1, H, W, 1) 

grid = torch.cat([col_space, row_space],axis=3)

sudo_right_img = torch.nn.functional.grid_sample(left_img, grid).squeeze(0)

sudo_right_img = transforms.functional.to_pil_image(sudo_right_img)

#plt.imshow(sudo_right_img)
#plt.show()

occlusion_mask = torch.ones(left_img.shape)
for row in range(H):
    max_val = -2
    for col in range(W):
        if col_space[:, row, col] <= max_val:
            occlusion_mask[:, :, row, col] = 0
        else:
            max_val = col_space[:, row, col]

for row in range(H):
    max_val = -2
		
    for col in range(W):
        if col_space[:, row, col] <= max_val:
            occlusion_mask[:, :, row, col] = 0
        else:
            max_val = col_space[:, row, col]


plt.imshow(sudo_right_img)
plt.show()













