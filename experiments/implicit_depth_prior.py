import matplotlib.pyplot as plt

import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.getcwd())

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64


# Load the left and right images
left_img_pil = load("data/input/left/tsukuba_daylight_L_00001.png")
right_img_pil = load("data/input/right/tsukuba_daylight_R_00001.png")
# left_img_pil = load("data/input/left/tsukuba_daylight_L_00200.png")
# right_img_pil = load("data/input/right/tsukuba_daylight_R_00200.png")

left_img_np = pil_to_np(left_img_pil)
right_img_np = pil_to_np(right_img_pil)

left_img_np = left_img_np[:3, :, :]
right_img_np = right_img_np[:3, :, :]

# temp get training mask from ground truth
mask = np.load("results/right_disp_mask.npy")

# visualize
# plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)

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
Debug_visualization = True
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

left_img_torch = torch.from_numpy(left_img_np).unsqueeze(0).type(dtype)
right_img_torch = torch.from_numpy(right_img_np).unsqueeze(0).type(dtype)

mask_torch = torch.from_numpy(mask).type(dtype)

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

    # need to fix the naming to make things more clear

    accuracy_weight = 0.60

    # left_right_disparity_consistency_loss = mse(warp(left_disparity, -right_disparity, dtype), right_disparity)
    # right_left_disparity_consistency_loss = mse(warp(right_disparity, left_disparity, dtype), left_disparity)
    #
    # consistency_loss = (left_right_disparity_consistency_loss + right_left_disparity_consistency_loss)/2

    # gamma = 10
    # right_disparity_occlusion_mask = torch.exp(-gamma * torch.abs(warp(left_disparity, -right_disparity, dtype) - right_disparity))
    # left_disparity_occlusion_mask = torch.exp(-gamma * torch.abs(warp(right_disparity, left_disparity, dtype) - left_disparity))

    right_disparity_occlusion_mask = torch.ones(left_disparity.shape).type(dtype) - \
                                     torch.abs(warp(left_disparity, -right_disparity, dtype) - right_disparity)
    left_disparity_occlusion_mask = torch.ones(right_disparity.shape).type(dtype) - \
                                    torch.abs(warp(right_disparity, left_disparity, dtype) - left_disparity)

    # convert to zero-one mask
    epsilon = 0.01

    right_disparity_occlusion_mask = torch.where(right_disparity_occlusion_mask < 1 - epsilon,
                                     torch.zeros_like(right_disparity_occlusion_mask),
                                     torch.ones_like(right_disparity_occlusion_mask))
    left_disparity_occlusion_mask = torch.where(left_disparity_occlusion_mask < 1 - epsilon,
                                    torch.zeros_like(left_disparity_occlusion_mask),
                                    torch.ones_like(left_disparity_occlusion_mask))



    right_consistency_mask = torch.ones(right_disparity.shape).type(dtype) - \
                             torch.abs(warp(right_img_torch, -right_disparity, dtype) - left_img_torch).mean(1)
    left_consistency_mask = torch.ones(left_disparity.shape).type(dtype) - \
                            torch.abs(warp(left_img_torch, left_disparity, dtype) - right_img_torch).mean(1)

    # convert to zero-one mask
    right_consistency_mask = torch.where(right_consistency_mask < 1 - epsilon,
                                                 torch.zeros_like(right_consistency_mask),
                                                 torch.ones_like(right_consistency_mask))
    left_consistency_mask = torch.where(left_consistency_mask < 1 - epsilon,
                                                torch.zeros_like(left_consistency_mask),
                                                torch.ones_like(left_consistency_mask))

    # Calculate consistency loss
    left_right_disparity_consistency_loss = mse(warp(left_disparity, -right_disparity, dtype) * right_consistency_mask,
                                                right_disparity * right_consistency_mask)
    right_left_disparity_consistency_loss = mse(warp(right_disparity, left_disparity, dtype) * left_consistency_mask,
                                                left_disparity * left_consistency_mask)

    consistency_loss = (left_right_disparity_consistency_loss + right_left_disparity_consistency_loss) / 2

    # Calculate accuracy loss
    left_disparity_loss = mse(warp(left_img_torch, left_disparity, dtype) * left_disparity_occlusion_mask,
                              right_img_torch * left_disparity_occlusion_mask)
    right_disparity_loss = mse(warp(right_img_torch, -right_disparity, dtype) * right_disparity_occlusion_mask,
                               left_img_torch * right_disparity_occlusion_mask)

    accuracy_loss = (left_disparity_loss + right_disparity_loss)/2

    # calculate total loss
    total_loss = (accuracy_loss * accuracy_weight) + (consistency_loss * (1 - accuracy_weight))
    # total_loss = (((left_disparity_loss + right_disparity_loss) * accuracy_weight)
    #               + ((left_right_disparity_consistency_loss + right_left_disparity_consistency_loss)
    #                  * (1 - accuracy_weight))) / 4

    # backprop
    total_loss.backward()

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

# out_np = torch_to_np(net(net_input))
# visualize the result
plot_image_grid([left_disparity_np, right_disparity_np], factor=5)
# save the result
np.save('results/implicit_depth_prior_left_result', left_disparity_np)
np.save('results/implicit_depth_prior_right_result', right_disparity_np)

# save the result for visualization
import cv2
visualiztion_output_left = ((left_disparity_np/np.max(left_disparity_np))*255).astype('uint8').squeeze()
visualiztion_output_right = ((right_disparity_np/np.max(right_disparity_np))*255).astype('uint8').squeeze()
cv2.imwrite('results/implicit_depth_prior_left_result.png', visualiztion_output_left)
cv2.imwrite('results/implicit_depth_prior_right_result.png', visualiztion_output_right)
