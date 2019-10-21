import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

def get_opencv_depth_images(left_images, right_images, focal_length=615., baseline=100):
    depth_images = []

    for L_path, R_path in zip(left_images, right_images):

        imgL = cv2.imread(L_path, 0)
        imgR = cv2.imread(R_path, 0)

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL,imgR)
        mask = disparity==0
        disparity[mask] = imgL.shape[0]
        min_disp = np.min(disparity)
        max_depth = baseline*focal_length/min_disp
        depth_image = baseline*focal_length / (disparity)
        depth_image[disparity == 0] = max_depth
        depth_images.append(depth_image)

    return depth_images

def depth_xml_to_images(depth_xml_paths):
    images = []
    for depth_xml_path in depth_xml_paths:
        cv_file = cv2.FileStorage(depth_xml_path, cv2.FILE_STORAGE_READ)
        image = cv_file.getNode("depth").mat()
        images.append(image)
    return images


if __name__=='__main__':


    focal_length = 615
    baseline = 100.
    left_images   = glob.glob('data/input/left/*')
    right_images  = glob.glob('data/input/right/*')
    left_images.sort()
    right_images.sort()
    depth_images = get_opencv_depth_images(left_images, right_images, focal_length=focal_length, baseline=baseline)
    ground_truths = depth_xml_to_images(['data/ground_truth/depth/tsukuba_depth_L_00001.xml'])

    plt.imshow(depth_images[0],'jet')
    plt.show()
    plt.imshow(ground_truths[0],'jet')
    plt.show()
