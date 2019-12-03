import matplotlib.pyplot as plt

import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.getcwd())
sys.path.append('../')

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim
import cv2
from utils.inpainting_utils import *
import math
def psnr(input_depth_image, target_depth_image):
    mse = np.mean( (input_depth_image - target_depth_image)** 2)
    return 20 * np.log10(1. / np.sqrt(mse))

def rmsel(img1, img2):
    return np.sqrt(((np.log(1 + img1)-np.log(1 + img2))**2).mean())
def ard(img1, img2):
    return (np.abs(img1 - img2)/(img2 + 1)).mean()
def srd(img1, img2):
    return (np.abs(img1 - img2)/((img2 + 1)**2)).mean()


def run_exp(exp_name, img_path, gnd_truth_path, mask_path, num_iter = 2000, use_mask=True, debug=False):

    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor

    PLOT = True
    imsize = -1
    dim_div_by = 64
    gnd_psnr_hist = []
    input_psnr_hist = []

    img_pil, img_np = get_image(img_path, imsize)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)
    numDisparities=64
    minDisparity=10

    _, gnd_np = get_image(gnd_truth_path, imsize) #cv2.imread(gnd_truth_path, 0)

    gnd_np = gnd_np[0, :, numDisparities + minDisparity:]
    img_mask_np = (img_mask_np==0).astype('float32')
    img_mask_pil = np_to_pil(img_mask_np)

    img_np      = pil_to_np(img_pil)

    img_mask_np = pil_to_np(img_mask_pil)

    img_mask_np = img_mask_np.astype('float32')

    img_mask_var = np_to_torch(img_mask_np).type(dtype)

    pad = 'reflection'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'

    # Build the network
    INPUT = 'noise'
    input_depth = 32
    LR = 0.01
    #num_iter = 2000
    param_noise = False
    show_every = 50
    figsize = 5
    #reg_noise_std = 0.03
    reg_noise_std = 0.08

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


    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)

    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)
    check_points = [0, 50, 100, 250, 500, 1000, 1500, 1700, 2000, 2200, 2500, 2700, 3000]
    max_ddip_pnsr = 0
    min_ddip_rmsel = 100000
    min_ddip_ard = 100000
    min_ddip_srd = 100000
    hc_pnsr = psnr(img_np, gnd_np)
    hc_rmsle = rmsel(img_np, gnd_np)
    hc_ard = ard(img_np, gnd_np)
    hc_srd = srd(img_np, gnd_np)
    if debug:
        print('ddip has to do better than: ', psnr(img_np, gnd_np))
        print()
    dir = '../results/ddip/' + exp_name + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
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
        #print(out.shape)
        # Calculate loss
        masked_total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss = mse(out, img_var)

        # backprop
        if use_mask:
            masked_total_loss.backward()
        else:
            total_loss.backward()

        #psnr_hist.append(psnr(out.cpu().data.numpy()[0][0], img_var.cpu().data.numpy()[0][0]))
        if i % 1 == 0:
            gnd_pnsr = psnr(out.cpu().data.numpy()[0][0], gnd_np)
            gnd_rmsel = rmsel(out.cpu().data.numpy()[0][0], gnd_np)
            gnd_ard = ard(out.cpu().data.numpy()[0][0], gnd_np)
            gnd_srd = srd(out.cpu().data.numpy()[0][0], gnd_np)

            if gnd_pnsr > max_ddip_pnsr:
                max_ddip_pnsr = gnd_pnsr
            if gnd_rmsel < min_ddip_rmsel:
                min_ddip_rmsel = gnd_rmsel
            if gnd_srd < min_ddip_srd:
                min_ddip_srd = gnd_srd
            if gnd_ard < min_ddip_ard:
                min_ddip_ard = gnd_ard


            input_psnr = psnr(out.cpu().data.numpy()[0][0], img_var.cpu().data.numpy()[0][0])
            gnd_psnr_hist.append(gnd_pnsr)
            input_psnr_hist.append(input_psnr)

        print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        if PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
        #         plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)

        # update the optimizer
        optimizer.step()
        if i in check_points:
            out_np = torch_to_np(net(net_input))
            visualiztion_output = (out_np*255).astype('uint8').squeeze()

            cv2.imwrite(dir + str(i) + '.png', visualiztion_output)

    out_np = torch_to_np(net(net_input))


    # visualize the result
    if debug:
        plot_image_grid([out_np], factor=5)
        # save the result
        noisy_psnr_hist = [hc_pnsr]*len(gnd_psnr_hist)
        plt.semilogx(np.arange(len(gnd_psnr_hist)), np.array(gnd_psnr_hist), label='Real Ground Truth')
        plt.semilogx(np.arange(len(input_psnr_hist)), np.array(input_psnr_hist), label = 'HC Ground Truth')
        plt.semilogx(np.arange(len(input_psnr_hist)), np.array(noisy_psnr_hist), label = 'Threshold')

        plt.xlabel('Iteration')
        plt.ylabel('PNSR(log)')
        plt.grid()
        plt.legend()
        plt.show()

    np.save(dir + exp_name + '.npy', out_np)

    visualiztion_output = (out_np*255).astype('uint8').squeeze()
    cv2.imwrite(dir + exp_name + '.png', visualiztion_output)

    return ([max_ddip_pnsr, min_ddip_rmsel, min_ddip_ard, min_ddip_srd], [hc_pnsr, hc_rmsle, hc_ard, hc_srd])

if __name__ =='__main__':

    ddip_result = []
    hc_result = []

    for i in range(10):
        print('\n Working on Image: ', i, '\n')
        exp_name = 'noisy_ddip_' + str(i) + '_'
        img_path = '../data/dan_new/noisy_' + str(i) + '.png'
        gnd_truth_path = '../data/dataset/groud_truth/' + str(i) + '.png'
        mask_path = '../data/dan_new/noisy_mask_' + str(i) + '.png'

        ddip, hc = run_exp(exp_name, img_path, gnd_truth_path, mask_path, num_iter = 1500, use_mask=True, debug=False)
        ddip_result.append(ddip)
        hc_result.append(hc)
    ddip_result = np.array(ddip_result)
    hc_result = np.array(hc_result)

    print()
    print('mean hc result: ', hc_result.mean(0), 'mean ddip result: ', ddip_result.mean(0))
#[2.37027076e+01 5.63142353e-03 3.78739311e-02 3.20355554e-02] mean ddip result:  [10.79985854  0.05423947  0.2215573   0.20163665]
