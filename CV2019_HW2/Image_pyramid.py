# Gaussian image pyramid
# Gaussian -> subsampling
# import cv2
import os
from os.path import splitext
import numpy as np
import math
from PIL import Image


# Gaussian filter
def Gaussian_filter(size=3, sigma=0.1):
    x, y = np.mgrid[-math.floor(size/2.):math.ceil(size/2.),
                    -math.floor(size/2.):math.ceil(size/2.)]
    # x, y = np.mgrid[-1:2, -1:2]

    gaussian_kernel = np.exp(-(x**2+y**2) / (2*sigma**2))

    # gaussian_kernel = np.exp(-(x**2+y**2))
    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel

# 2D Convolution


def Conv2d(W, F, S, P):
    # define shape of output
    output_shape = (np.array(W.shape, dtype=int) -
                    np.array(F.shape, dtype=int) + 2*P)/S + 1
    sum_filter = 0
    output_int = np.zeros(
        (int(output_shape[0]), int(output_shape[1])), dtype=np.int32)
                                                 # note dtype int != np.int32
    print("Input shape is ", W.shape)
    print("Output shape is ", output_int.shape)
    for i in range(int(output_shape[0])):
        for j in range(int(output_shape[1])):
            sum_filter = 0
            for k in range(F.shape[0]):
                for m in range(F.shape[1]):
                    sum_filter += F[k][m] * W[i+k][j+m]
            output_int[i][j] = sum_filter
    return output_int


def GCD(a, b):
    m = max(a, b)
    n = min(a, b)
    while True:
        c = m % n
        if c == 0:
            return n
        m = n
        n = c
    return c

# Subsampling / downsampling


def Subsampling(image, s):
    output = np.zeros(
        (int(image.shape[0]/s), int(image.shape[1]/s)), dtype=np.int32)
    tmp = 0
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            tmp = 0
            for k in range(s):
                for m in range(s):
                    tmp += image[s*i+k][s*j+m]
            output[i][j] = tmp/(s**2)

    return output

# Image Pyramid


def Image_Pyramid(layer, image, smooth, filename, kernel):
    filename = splitext(filename)[0] + ".png"
    print(filename)
    for i in range(layer):
        print("Layer", i)
        if (image.shape[0] <= kernel.shape[0] or image.shape[1] <= kernel.shape[1]):
            break
        if smooth == True:
            image = Conv2d(image, kernel, 1, 0)
        if (image.shape[0] <= 2 or image.shape[1] <= 2):
            break
        image = Subsampling(image, 2)
        if smooth == True:
            Image.fromarray(image, 'I').convert('L').save("kernel_"+str(kernel.shape[0])+"/s_"+str(i)+"_"+filename)
        else:
            Image.fromarray(image, 'I').convert('L').save(str(i)+"_"+filename)


# Open images in data folder
path = 'hw2_data/task1and2_hybrid_pyramid/'
images = []
gf = Gaussian_filter(size=3, sigma=0.33)
i = 0

for filename in os.listdir(path):
    i+=1
    if (filename == '.DS_Store'):
        continue
    fn = path + filename
    images.append(np.array(Image.open(fn).convert("I")))
    Image_Pyramid(8, images[-1], True, filename, gf)
    print("Finish", filename, "'s image pyramid.")