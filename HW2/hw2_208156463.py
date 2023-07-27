import cv2
import matplotlib.pyplot as plt
import numpy as np

# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )


def find_transform(pointset1, pointset2):

   # np.set_printoptions(suppress=True)
    T = np.zeros((3,3), dtype=np.float32)
    T_vector = np.zeros((8,1), dtype=np.float32)
    number_of_points = pointset1.shape[0]
    X = np.zeros((2*number_of_points, 8), dtype=np.float32)

    i = 0
    row = 0
    while row < X.shape[0]:
        arr1 = np.array([pointset1[i][0],pointset1[i][1],0,0,1,0,(-1 * pointset1[i][0] * pointset2[i][0]),(-1 * pointset1[i][1] * pointset2[i][0])])
        X[row] = arr1
        arr2 = np.array([0,0,pointset1[i][0],pointset1[i][1],0,1, (-1 * pointset1[i][0] * pointset2[i][1]),(-1 * pointset1[i][1] * pointset2[i][1])])
        X[row+1] = arr2
        i = i + 1
        row = row+2

    pointset2_vector = np.zeros((2*number_of_points, 1), dtype=np.float32)
    row=0
    i=0
    while row < pointset2_vector.shape[0]:
        pointset2_vector[row] = pointset2[i][0]
        pointset2_vector[row+1] = pointset2[i][1]
        i = i+1
        row = row+2

    X_inverst = np.linalg.pinv(X)
    T_vector = np.matmul(X_inverst,pointset2_vector)

    T[0] = np.array([T_vector[0][0], T_vector[1][0], T_vector[4][0]]) #a,b,e
    T[1] = np.array([T_vector[2][0], T_vector[3][0], T_vector[5][0]]) #c,d,f
    T[2] = np.array([T_vector[6][0], T_vector[7][0], 1]) #g,h,1

    return T


def trasnform_image(image, T):

    new_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    T_inverst = np.linalg.pinv(T)
    new_pixel_vector = np.zeros((3, 1), dtype=np.float32)

    for y in range(new_image.shape[0]):
        for x in range(new_image.shape[1]):
            new_pixel_vector[0][0] = x
            new_pixel_vector[1][0] = y
            new_pixel_vector[2][0] = 1
            original_pixel_vector = np.matmul(T_inverst, new_pixel_vector)
            x_original = round(original_pixel_vector[0][0]/original_pixel_vector[2][0])
            y_original = round(original_pixel_vector[1][0]/original_pixel_vector[2][0])
            if 0 <= x_original <= n-1 and 0 <= y_original <= m-1:
                new_image[y][x] = image[y_original][x_original]

    return new_image


def create_wormhole(im, T, iter=5):

    new_image = np.zeros((im.shape[0], im.shape[1]))
    new_image = im
    for i in range(iter):
        new_image = trasnform_image(new_image, T)
        im = np.clip(im + new_image, 0, 255)

    return im

