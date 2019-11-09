import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

def get_opencv_depth_images(left_images,
                            right_images,
                            focal_length=615.,
                            baseline=100,
                            numDisparities=32,
                            blockSize=13,
                            minDisparity=10):
    depth_images = []
    masks = []
    disparities = []
    index = 0
    for L_path, R_path in zip(left_images, right_images):
        print(L_path, R_path)
        imgL = cv2.resize(cv2.imread(L_path, 0), (384, 288))
        imgR = cv2.resize(cv2.imread(R_path, 0), (384, 288))
        stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blockSize)
        disparity = stereo.compute(imgL,imgR)
        disparity = disparity[:, numDisparities + minDisparity:]

        disparity = disparity.astype('float32')/16
        mask = disparity==(minDisparity - 1)
        np.save('data/dataset/disparity/%d.npy'%(index), disparity)
        np.save('data/dataset/mask/%d.npy'%(index), mask.astype('uint8'))

        disparity -= disparity.min()
        disparity /= disparity.max()

        disparity *= 255
        masks.append(mask)
        disparities.append(disparity)

        cv2.imwrite('data/opencv-5-64/%d.png'%(index), disparity)
        cv2.imwrite('data/opencv-5-64/mask_%d.png'%(index), 255*mask.astype('uint8'))

        index += 1
    return masks, disparities

def depth_xml_to_images(depth_xml_paths):
    images = []
    for depth_xml_path in depth_xml_paths:
        cv_file = cv2.FileStorage(depth_xml_path, cv2.FILE_STORAGE_READ)
        image = cv_file.getNode("depth").mat()
        images.append(image)
def psnr(input_depth_image, target_depth_image):
    mse = np.mean( (input_depth_image - target_depth_image)** 2)
    return 20 * np.log(255.0 / np.sqrt(mse))


if __name__=='__main__':

#    ground_truth_image = cv2.imread('data/dataset/groud_truth/0.png', 0)[:, 42:]
#    print('g: ', ground_truth_image.shape)
#    target_image = cv2.imread('data/dataset/disparity/0.png', 0)
#    print('target: ', target_image.shape)
#    output_image = cv2.resize(cv2.imread('results/inpainting.png', 0)
#    print('out: ', output_image.shape)
#    print(psnr(target_image, ground_truth_image))
#    print(psnr(output_image, ground_truth_image))



    focal_length = 615
    baseline = 100.0
    left_images   = glob.glob('data/input/left/*') #['data/tsukuba_l.png']#
    right_images  = glob.glob('data/input/right/*')#['data/tsukuba_r.png']#
    left_images.sort()
    right_images.sort()
    block_sizes = [5]#[5, 11, 17, 21]
    n_disparities =[64]# [16, 32, 64]
    min_disparities = [0, 10, 20]
    for bs in block_sizes:
        for nd in n_disparities:
            print('I am using % d blockSize and %d number of disparities'%(bs, nd))
            masks, disparities = get_opencv_depth_images(left_images,
                                                         right_images,
                                                         focal_length=focal_length,
                                                         baseline=baseline,
                                                         numDisparities=nd,
                                                         blockSize=bs)
