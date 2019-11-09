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
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64

# Get image path
img_path  = 'data/dataset/disparity/0.png'#'data/inpainting/kate.png'
# Get mask path
mask_path = 'data/dataset/mask/mask_0.png'#'data/inpainting/kate_mask.png'

# Get image
img_pil, img_np = get_image(img_path, imsize)
# get mask
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

img_mask_np = (img_mask_np==0).astype('float32')
img_mask_pil = np_to_pil(img_mask_np)
# Center_crop the image and mask
# img_mask_pil = crop_image(img_mask_pil, dim_div_by)
# img_pil      = crop_image(img_pil,      dim_div_by)

img_np      = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

# Make the mask a torch tensor
img_mask_var = np_to_torch(img_mask_np).type(dtype)

# visualize
plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)

# Training params
pad = 'reflection'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

# Build the network
INPUT = 'noise'
input_depth = 32
LR = 0.01
num_iter = 1 #6001
param_noise = False
show_every = 50
figsize = 5
reg_noise_std = 0.03

net = skip(input_depth, 1,
           num_channels_down = [128] * 5,
           num_channels_up =   [128] * 5,
           num_channels_skip =    [128] * 5,
           filter_size_up = 3, filter_size_down = 3,
           upsample_mode='nearest', filter_skip_size=1,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)



net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)

# i = 0
# def closure():
#
#     global i
#
#     if param_noise:
#         for n in [x for x in net.parameters() if len(x.size()) == 4]:
#             n = n + n.detach().clone().normal_() * n.std() / 50
#
#     net_input = net_input_saved
#     if reg_noise_std > 0:
#         net_input = net_input_saved + (noise.normal_() * reg_noise_std)
#
#
#     out = net(net_input)
#
#     total_loss = mse(out * mask_var, img_var * mask_var)
#     total_loss.backward()
#
#     print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
#     if  PLOT and i % show_every == 0:
#         out_np = torch_to_np(out)
# #         plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
#
#     i += 1
#
#     return total_loss

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)

print('Starting optimization with ADAM')
optimizer = torch.optim.Adam(p, lr=LR)

for i in range(num_iter):
    optimizer.zero_grad()

    # add noise to network parameters
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    # add noise to network input
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    # get network output
    out = net(net_input)

    # Calculate loss
    total_loss = mse(out * mask_var, img_var * mask_var)
    # backprop
    total_loss.backward()

    # visualize
    print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
    #         plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)

    # update the optimizer
    optimizer.step()

# optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = torch_to_np(net(net_input))
# visualize the result
plot_image_grid([out_np], factor=5)
# save the result
np.save('results/testing_inpainting_V2_result', out_np)

# print(out_np.shape)
# import cv2
# visualiztion_output = ((out_np/np.max(out_np))*255).astype('uint8')
# cv2.imwrite('results/testing_inpainting_V2_result.png', visualiztion_output)
