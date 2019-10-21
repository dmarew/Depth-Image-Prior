import numpy as np
import cv2
from matplotlib import pyplot as plt

def get_depth_image(imgL, imgR, focal_length=615., baseline=100):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    mask = disparity==0
    disparity[mask] = imgL.shape[0]
    min_disp = np.min(disparity)
    max_depth = baseline*focal_length/min_disp
    depth_image = baseline*focal_length / (disparity)
    depth_image[disparity == 0] = max_depth

    return depth_image

if __name__=='__main__':

    imgL = cv2.imread('data/tsukuba_l.png',0)
    imgR = cv2.imread('data/tsukuba_r.png',0)
    focal_length = 615
    baseline = 100.
    depth_image = get_depth_image(imgL, imgR, focal_length=focal_length, baseline=baseline)
    print(depth_image)
    plt.imshow(depth_image,'jet')
    plt.show()
