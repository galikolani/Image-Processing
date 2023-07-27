import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2


def clean_baby(im):
    h,w = im.shape
    coords_img = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    baby_coords1 = np.float32([[6, 20], [111, 20], [111, 130], [6, 130]])

    baby_coords2 = np.float32([[77, 162], [147, 117], [245, 160], [133, 245]])

    baby_coords3 = np.float32([[180, 6], [250, 70], [176, 120], [120, 51]])

    t1 = cv2.getPerspectiveTransform(baby_coords1, coords_img)
    t2 = cv2.getPerspectiveTransform(baby_coords2, coords_img)
    t3 = cv2.getPerspectiveTransform(baby_coords3, coords_img)

    dst1 = cv2.warpPerspective(im, t1, (h,w)).astype('uint8')
    dst2 = cv2.warpPerspective(im, t2, (h,w)).astype('uint8')
    dst3 = cv2.warpPerspective(im, t3, (h,w)).astype('uint8')

    img_stack = np.array([dst1, dst2, dst3])
    res = np.median(img_stack, axis=0).astype('uint8')
    clean_im = cv2.medianBlur(res, ksize=7)

    return clean_im

def clean_windmill(im):

    img_fourier = np.fft.fft2(im)
    img_fourier = np.fft.fftshift(img_fourier)

    # we found the peak pixels in the fourier image that represent the waves in the image - so we zeroing them
    img_fourier[124][100] = 0
    img_fourier[132][156] = 0

    img_fourier = np.fft.ifftshift(img_fourier)
    img_inv = np.fft.ifft2(img_fourier)
    clean_im = np.abs(img_inv)

    return clean_im

def clean_watermelon(im):

    mask = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])
    clean_im = convolve2d(im, mask, mode='same')
    return clean_im

def clean_umbrella(im):

    img_fourier = np.fft.fft2(im)
    mask = np.zeros((im.shape[0], im.shape[1]))

    # pixel[y][x] moves to [y+5][x+80] so we make a mask to fix it
    mask[1][1] = 0.5
    mask[5][80] = 0.5

    mask_fourier = np.fft.fft2(mask)
    indexZero = np.where(np.abs(mask_fourier) < 0.000001) # pixels that are very small we change them to 1
    mask_fourier[indexZero] = 1
    clean_im = np.divide(img_fourier, mask_fourier)
    clean_im[indexZero] = 0                             # change back to 0 the pixels that was very small
    inverse_clean_im = np.fft.ifft2(clean_im)
    clean_im = np.abs(inverse_clean_im)

    return clean_im

def clean_USAflag(im):
    clean_im = np.zeros((im.shape[0],im.shape[1]))

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            clean_im[i][j] = im[i][j]

    for i in range(10,im.shape[1]-9):
        for j in range(90, 168):
            mask = im[j, i - 9:i + 9]
            med = np.median(mask)
            clean_im[j][i] = med

    for i in range(154,300-9):
        for j in range(1, 90):
            mask = im[j, i - 9:i + 9]
            med = np.median(mask)
            clean_im[j][i] = med

    return clean_im

def clean_cups(im):

    img_fourier = np.fft.fft2(im)
    mask = np.ones((256,256))
    # we found a noise  that looks like square in the fourier image that represents the noise in the image
    mask[109:149,109:149] = 1.5
    mask_shift = np.fft.fftshift(mask)
    clean_im = np.fft.ifft2(img_fourier*mask_shift)
    clean_im = np.abs(clean_im)
    return clean_im

def clean_house(im):

    mask = np.zeros((im.shape[0], im.shape[1]))

    for i in range(10):
        mask[0][i] = 1 / 10

    img_fourier = np.fft.fft2(im)
    mask_fourier = np.fft.fft2(mask)

    mask_fourier_comp = np.conjugate(mask_fourier)
    clean_im = img_fourier * (mask_fourier_comp/( (mask_fourier*mask_fourier_comp)+  0.00125))

    inverse_clean_im = np.fft.ifft2(clean_im)
    clean_im = np.abs(inverse_clean_im)

    return clean_im

def clean_bears(im):

    #adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    clean_im = clahe.apply(im)

    return clean_im

'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''