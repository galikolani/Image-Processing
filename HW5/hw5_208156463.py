import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt


def sobel(im):
    subel_X = (1/8) * np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    subel_Y = (1/8) * np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

    new_imx = conv(im, subel_X, mode='same')
    new_imy = conv(im, subel_Y, mode='same')
    new_im = abs(new_imx) + abs(new_imy)

    threshold = 17
    res,new_im = cv2.threshold(new_im, threshold,255, cv2.THRESH_BINARY)

    return new_im


def canny(im):
    im_blur = gaussian(im, 1.6)
    t_lower = 60
    t_upper = 240

    return cv2.Canny(im_blur, t_lower, t_upper)


def hough_circles(im):
    im_c = im.copy()
    circles = cv2.HoughCircles(im_c, cv2.HOUGH_GRADIENT, 1.3, 50, param1=70, param2=50, minRadius=30, maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(im_c, center, radius, (0, 0, 0), 2)
    return im_c


def hough_lines(im):
    im_l = im.copy()

    im_l_gaussed = gaussian(im.copy(), 1.5)
    im_l_c = cv2.Canny(im_l_gaussed, 80, 400)
    lines = cv2.HoughLines(im_l_c, 1, np.pi / 180, 150)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(im_l, pt1, pt2, (0, 0, 0), 3, cv2.LINE_AA)

    return im_l

if __name__=='__main__':
    im = cv2.imread(r'balls1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sob = sobel(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(sob, cmap='gray', vmin=0,vmax=255)
    plt.show()


    im = cv2.imread(r'coins1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    can = canny(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(can, cmap='gray')
    plt.show()

    im = cv2.imread(r'coins3.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    houghc = hough_circles(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(houghc, cmap='gray')
    plt.show()

    im = cv2.imread(r'boxOfchocolates1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    houghl = hough_lines(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(houghl, cmap='gray')
    plt.show()