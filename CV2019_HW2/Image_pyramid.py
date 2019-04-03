# Gaussian image pyramid
# Gaussian -> subsampling
import cv2
import numpy as np
import math
# Gaussian filter


def Gaussian_filter(size=3, sigma=math.sqrt(2)):
    x, y = np.mgrid[-math.floor(size/2.):math.ceil(size/2.),
                    -math.floor(size/2.):math.ceil(size/2.)]
    # x, y = np.mgrid[-1:2, -1:2]

    gaussian_kernel = np.exp(-(x**2+y**2) / (2*sigma**2))

    # gaussian_kernel = np.exp(-(x**2+y**2))
    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel


gf = Gaussian_filter()
print(gf)
