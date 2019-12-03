import matplotlib.pyplot as plt

import os
import sys
import subprocess
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.getcwd())

# import numpy as np
# from models.resnet import ResNet
# from models.unet import UNet
# from models.skip import skip
# import torch
# import torch.optim
#
# from utils.inpainting_utils import *

image_numbers = ["00001", "00200", "00400", "00600", "00800"]
stds = [0.0, 0.005, 0.01]

print("number of runs:", len(image_numbers) * len(stds))
for inum in image_numbers:
    for std in stds:
        left_img = "data/input/left/tsukuba_daylight_L_" + inum + ".png"
        right_img = "data/input/right/tsukuba_daylight_R_" + inum + ".png"

        left_disp_img = "data/ground_truth/disparity/left/tsukuba_disparity_L_" + inum + ".png"
        right_disp_img = "data/ground_truth/disparity/right/tsukuba_disparity_R_" + inum + ".png"

        command_line = "experiments/implicit_depth_prior.py" \
                       + " --left_image=" + left_img \
                       + " --right_image=" + right_img \
                       + " --left_disp_image=" + left_disp_img \
                       + " --right_disp_image=" + right_disp_img \
                       + " --noise=" + str(std)

        print("executing:", command_line)
        print()

        subprocess.call(command_line, shell=True)
        # exec(open(command_line).read())

print("done!!")