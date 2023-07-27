import numpy as np
import matplotlib.pyplot as plt
import cv2


def histImage(im):

     h = np.zeros(256, dtype=int)
     for row in range(im.shape[0]):
         for col in range(im.shape[1]):
             h[im[row][col]] = h[im[row][col]] + 1
     return h


def nhistImage(im):

    h = histImage(im)
    nh = np.zeros(256)
    
    for index, val in enumerate(h):
        nh[index] = val/im.size
    
    return nh


def ahistImage(im):

    his = histImage(im)
    ah = np.cumsum(his)
    return ah


def calcHistStat(h):

    m = sum([index*val for index , val in enumerate(h)]) / h.sum()
    e = (sum([(index**2)*val for index , val in enumerate(h)]) / h.sum()) - m**2
    return m, e


def mapImage(im,tm):

    nim = np.clip([tm[i] for i in im], 0, 255)
    return nim

def histEqualization(im):

    sumPix = im.size
    avr = sumPix/256
    ahist=ahistImage(im)
    tm = np.zeros(256, dtype=int)
    helpAcc = np.zeros(256, dtype=int) + avr
    helpAcc = np.cumsum(helpAcc)

    i=0
    j=0
    while i<=255 and j<=255:
        if ahist[i] <= helpAcc[j]:
            tm[i] = j
            i += 1
        else:
            j += 1

    return tm




