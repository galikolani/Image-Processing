import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()
    H, W = im.shape
    pixels = np.arange(start=0,stop=H*W,step=1)
    pixels = pixels.tolist()
    sample = random.sample(pixels, round(H*W*p))

    for i in range(len(sample)):
        row = int(sample[i]/H)
        col = int(sample[i]%H)
        if i%2 == 0:
            sp_noise_im[row][col] = 0
        else:
            sp_noise_im[row][col] = 255


    return sp_noise_im


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    H, W = im.shape
    clean_im = np.zeros((H,W))
    size_matrix_median = radius * 2 + 1
    matrix_median = np.zeros((size_matrix_median, size_matrix_median))
    median = 0

    for i in range(H):
        for j in range(W):
            matrix_median[int(size_matrix_median / 2)][int(size_matrix_median / 2)] = noise_im[i][j]
            X, Y = np.meshgrid(np.arange(i - radius, i + radius + 1), np.arange(j - radius, j + radius + 1))

            for row in range(size_matrix_median):
                for col in range(size_matrix_median):
                    if X[row][col] < 0 or X[row][col] >= H:
                        matrix_median[row][col] = 0
                    elif Y[row][col] < 0 or Y[row][col] >= W:
                        matrix_median[row][col] = 0
                    else:
                        matrix_median[row][col] = noise_im[X[row][col]][Y[row][col]]

            median = np.median(matrix_median)
            clean_im[i][j] = median



    return clean_im


def clean_SP_noise_multiple(images):
    # TODO: add implementation
    clean_image = np.median(images, axis=0)

    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    H, W = im.shape
    normal = np.random.normal(0,s, size=(H,W))
    gaussian_noise_im = im + normal
    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):

    H, W = im.shape
    cleaned_im = np.zeros((H, W))
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size,kernel_size))

    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            kernel[x][y] = np.exp( -(x**2 + y**2)/(2*maskSTD**2) )

    gaussian_matrix_radius = kernel / np.sum(kernel)
    cleaned_im = convolve2d(im, gaussian_matrix_radius, mode="same")

    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if (i>0 and j>0) and (i<im.shape[0]-1 and j<im.shape[1]-1):
                X, Y = np.meshgrid(np.arange(i - radius, i + radius + 1), np.arange(j - radius, j + radius + 1))
                window = im[i-radius:i+radius+1, j-radius:j+radius+1]
                gi = np.exp( -((window - im[i][j])**2) / (2*stdIntensity**2) )
                gs = np.exp( -((X-i)**2 + (Y-j)**2) / (2*stdSpatial**2) )

                sum1 = np.sum(gi * gs * window)
                sum2 = np.sum(gi * gs)
                bilateral_im[i][j] = np.clip(sum1/sum2, 0, 255)

    return bilateral_im.astype(np.uint8)
