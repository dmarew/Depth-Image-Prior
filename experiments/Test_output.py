import sys, os
from PIL import Image
import numpy as np
import cv2
sys.path.append(os.getcwd())
from utils.inpainting_utils import *
HC_left = np.asarray(Image.open('data/dataset/disparity/0.png'))
DDIP = np.asarray(Image.open('results/testing_inpainting_V2_result.png'))
IDIP = np.asarray(Image.open('results/implicit_depth_prior_result.png'))
left_gt = cv2.resize(cv2.imread('data/dataset/groud_truth/0.png', 0), (384, 288))[:, 74:]
right_gt = cv2.imread('data/ground_truth/disparity/right/tsukuba_disparity_R_00001.png', 0)
W, H = DDIP.shape
#left_gt = left_gt/255#left_gt.shape
#right_gt = right_gt/255

psnr_HC_left   = psnr(HC_left, left_gt)
print(psnr_HC_left)
psnr_DDIP = psnr(DDIP, left_gt)
psnr_IDIP = psnr(IDIP, right_gt)

print(psnr_HC_left, psnr_DDIP, psnr_IDIP)