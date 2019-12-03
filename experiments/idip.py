#!/usr/bin/env python3

import matplotlib.pyplot as plt

import os
import sys
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.getcwd())

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from utils.inpainting_utils import *

# Take in command line arguments
parser = argparse.ArgumentParser(description='Reconstruction using deep implicit prior.')
parser.add_argument("--left_image", type=str, help="Path to the left image", default="data/input/left/tsukuba_daylight_L_00001.png")
parser.add_argument("--right_image", type=str, help="Path to the left image", default="data/input/right/tsukuba_daylight_R_00001.png")

parser.add_argument("--left_disp_image", type=str, help="Path to the left disparity image", default="data/ground_truth/disparity/left/tsukuba_disparity_L_00001.png")
parser.add_argument("--right_disp_image", type=str, help="Path to the left disparity image", default="data/ground_truth/disparity/right/tsukuba_disparity_R_00001.png")

parser.add_argument("--noise", type=float, help="standard deviation of added noise", default=0)

args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64


def gaussian_noise(img, std):
    gauss = np.random.normal(0, std, img.shape)
    noisy_img = np.clip(img + gauss, 0, 1)
    return noisy_img


# Load the left and right images
left_img_pil = load(args.left_image)
right_img_pil = load(args.right_image)

left_img_np = pil_to_np(left_img_pil)
right_img_np = pil_to_np(right_img_pil)

left_img_np = left_img_np[:3, :, :]
right_img_np = right_img_np[:3, :, :]

# add noise
left_img_np = gaussian_noise(left_img_np, args.noise)
right_img_np = gaussian_noise(right_img_np, args.noise)

# visualize the noisy images
# plot_image_grid([left_img_np, right_img_np], factor=5, nrow=1)

gt_left_disp_img_pil = load(args.left_disp_image)
gt_right_disp_img_pil = load(args.right_disp_image)

gt_left_disp_img_np = pil_to_np(gt_left_disp_img_pil)
gt_right_disp_img_np = pil_to_np(gt_right_disp_img_pil)

# Training params
pad = 'reflection'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

# Build the network
INPUT = 'noise'
input_depth = 32
LR = 0.01
num_iter = 6001
param_noise = False
Debug_visualization = False
show_every = 1000
figsize = 5
reg_noise_std = 0.05 # 0.03

# there is probably an easy way to copy this
net = skip(input_depth, 2,
           num_channels_down = [128] * 5,
           num_channels_up =   [128] * 5,
           num_channels_skip =    [128] * 5,
           filter_size_up = 3, filter_size_down = 3,
           upsample_mode='nearest', filter_skip_size=1,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)


net = net.type(dtype)

net_input = get_noise(input_depth, INPUT, left_img_np.shape[1:]).type(dtype)
net_right_input = get_noise(input_depth, INPUT, right_img_np.shape[1:]).type(dtype)

# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

# convert to torch
left_img_torch = torch.from_numpy(left_img_np).unsqueeze(0).type(dtype)
right_img_torch = torch.from_numpy(right_img_np).unsqueeze(0).type(dtype)

gt_left_disp_img_torch = torch.from_numpy(gt_left_disp_img_np).type(dtype)
gt_right_disp_img_torch = torch.from_numpy(gt_right_disp_img_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

net_right_input_saved = net_right_input.detach().clone()
noise_right = net_right_input.detach().clone()

# need to figure out this part
p = get_params(OPT_OVER, net, net_input)

print('Starting optimization with ADAM')
optimizer = torch.optim.Adam(p, lr=LR)


# define the warp function
def warp(input_img, disp_img, dtype):
    B, C, H, W = input_img.shape
    row_space = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W).unsqueeze(0).type(dtype)
    col_space = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1).unsqueeze(0).type(dtype)

    col_space += disp_img.squeeze()

    grid = torch.zeros((1, H, W, 2)).type(dtype)
    grid[:, :, :, 1] = row_space
    grid[:, :, :, 0] = col_space
    output_img = torch.nn.functional.grid_sample(input_img, grid)

    return output_img


def diff_mask(img1, img2, epsilon, dtype):
    mask = torch.ones(img1.shape).type(dtype) - torch.abs(img1 - img2)
    if len(mask.shape) == 4:
        mask = mask.mean(1).unsqueeze(0)
    mask = torch.where(mask < 1 - epsilon,
                       torch.zeros_like(mask),
                       torch.ones_like(mask))
    return mask


data_log = np.zeros([5, num_iter])

for i in range(num_iter):
    optimizer.zero_grad()

    # Add noise to network parameters
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    # Add noise to network input
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    # Get network output
    out = net(net_input)
    left_disparity = out[:, 0, :, :].unsqueeze(1)
    right_disparity = out[:, 1, :, :].unsqueeze(1)

    # Note left disparity tells me how to map the right image to look like the left image
    # Similarly right disparity tells me how to map the left image to look like the right image

    accuracy_weight = 0.40
    epsilon = 0.01

    # Create occlusion mask
    right_disparity_occlusion_mask = diff_mask(warp(left_disparity, right_disparity, dtype), right_disparity, epsilon, dtype)
    left_disparity_occlusion_mask = diff_mask(warp(right_disparity, -left_disparity, dtype), left_disparity, epsilon, dtype)

    # Create consistency mask
    right_consistency_mask = diff_mask(warp(left_img_torch, right_disparity, dtype), right_img_torch, epsilon, dtype)
    left_consistency_mask = diff_mask(warp(right_img_torch, -left_disparity, dtype), left_img_torch, epsilon, dtype)

    # Calculate consistency loss
    left_right_disparity_consistency_loss = mse(warp(left_disparity, right_disparity, dtype) * right_consistency_mask,
                                                right_disparity * right_consistency_mask)
    right_left_disparity_consistency_loss = mse(warp(right_disparity, -left_disparity, dtype) * left_consistency_mask,
                                                left_disparity * left_consistency_mask)

    consistency_loss = (left_right_disparity_consistency_loss + right_left_disparity_consistency_loss) / 2

    # Calculate accuracy loss
    left_disparity_loss = mse(warp(right_img_torch, -left_disparity, dtype) * left_disparity_occlusion_mask,
                              left_img_torch * left_disparity_occlusion_mask)
    right_disparity_loss = mse(warp(left_img_torch, right_disparity, dtype) * right_disparity_occlusion_mask,
                               right_img_torch * right_disparity_occlusion_mask)

    accuracy_loss = (left_disparity_loss + right_disparity_loss)/2

    # calculate total loss
    total_loss = (accuracy_loss * accuracy_weight) + (consistency_loss * (1 - accuracy_weight))

    # backprop
    total_loss.backward()

    # Calculate the ground truth loss
    gt_left_loss = mse(gt_left_disp_img_torch, left_disparity)
    gt_right_loss = mse(gt_right_disp_img_torch, right_disparity)

    # log the data
    data_log[:, i] = [accuracy_loss.item(), consistency_loss.item(), total_loss.item(), gt_left_loss, gt_right_loss]

    # visualize
    print('Iteration %05d    Accuracy Loss %f, Consistency Loss %f, Total Loss %f' %
          (i, accuracy_loss.item(), consistency_loss.item(), total_loss.item()), '\r', end='')
    if PLOT and i % show_every == 0 and Debug_visualization:
        left_disparity_np = torch_to_np(left_disparity)
        right_disparity_np = torch_to_np(right_disparity)
        left_mask_np = torch_to_np(left_disparity_occlusion_mask)
        right_mask_np = torch_to_np(right_disparity_occlusion_mask)
        left_con_mask_np = torch_to_np(left_consistency_mask)
        right_con_mask_np = torch_to_np(right_consistency_mask)
        plot_image_grid([np.clip(left_disparity_np, 0, 1),
                         np.clip(right_disparity_np, 0, 1),
                         np.clip(left_mask_np, 0, 1),
                         np.clip(right_mask_np, 0, 1),
                         np.clip(left_con_mask_np, 0, 1),
                         np.clip(right_con_mask_np, 0, 1)], factor=figsize, nrow=2)

    # update the optimizer
    optimizer.step()

out = net(net_input)
left_disparity = out[:, 0, :, :].unsqueeze(1)
right_disparity = out[:, 1, :, :].unsqueeze(1)

left_disparity_np = torch_to_np(left_disparity)
right_disparity_np = torch_to_np(right_disparity)

# visualize the result
if Debug_visualization:
    plot_image_grid([left_disparity_np, right_disparity_np], factor=5)
# save the result
filename = "results/implicit_depth_prior_" + str(args.noise) + "_" + args.left_image[len(args.left_image) - 8:len(args.left_image) - 4] + "_"

# np.save(filename + "left_result", left_disparity_np)
# np.save(filename + "right_result", right_disparity_np)
# np.save(filename + "data_log", data_log)
#
# # save the result for visualization
# import cv2
# visualiztion_output_left = ((left_disparity_np/np.max(left_disparity_np))*255).astype('uint8').squeeze()
# visualiztion_output_right = ((right_disparity_np/np.max(right_disparity_np))*255).astype('uint8').squeeze()
# cv2.imwrite(filename + "left_result.png", visualiztion_output_left)
# cv2.imwrite(filename + "right_result.png", visualiztion_output_right)
