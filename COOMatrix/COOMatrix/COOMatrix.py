import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.exposure as exposure
import math
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage.io import imread
from skimage.feature import greycomatrix
format = 'F:/studies/diploma/DataSets-Computer-Vision/data.brodatz/size_128x128/D{:03d}_{:02d}.png'
classesStart = 1
classesEnd = 112

offset = 5

imgStart = 1
imgEnd = 10

def get_image(cls, img) :
    imgName = format.format(cls, img)
    image=imread(imgName)
    return image

def comatrix (image) :
    result = greycomatrix(image, [offset], [0], 256, symmetric=True, normed=True)
    return result

def imageCOML1 (image1, image2) :
    arr1 = comatrix(image1)
    arr2 = comatrix(image2)
    np.reshape(arr1, 256*256)
    np.reshape(arr2, 256*256)
    result = abs(arr1 - arr2)
    return np.sum(result)

def calculate_distances(clS, clF, imgS, imgF) :
    result = np.empty((clF - clS, imgF - imgS, clF - clS, imgF - imgS))
    for i in range(clS, clF) :
        if (i == 14) :
            continue #image is absent in dataset
        for j in range (imgS, imgF) :
            for i1 in range(clS, clF) :
                if (i1 == 14) :
                    continue #image is absent in dataset
                for j1 in range (imgS, imgF) :
                    result[i-clS][j-imgS][i1-clS][j1-imgS] = imageCOML1(get_image(i, j), get_image(i1, j1))
    return result

correct = 0
all = 1. * (imgEnd - imgStart) * (classesEnd - classesStart - 1) #1 for the 14-th class
distances = calculate_distances(classesStart, classesEnd, imgStart, imgEnd)
for i in range (classesStart, classesEnd) :
    if (i == 14) :
        continue #image is absent in dataset
    for j in range (imgStart, imgEnd) :
        cur_dist = 1000000.
        cur_class = -1
        for i1 in range (classesStart, classesEnd) :
            if (i1 == 14) :
                continue #image is absent in dataset
            for j1 in range (imgStart, imgEnd) :
                if (i == i1 and j == j1) :
                    continue
                if (cur_dist > distances[i-classesStart][j-imgStart][i1-classesStart][j1-imgStart] and distances[i-classesStart][j-imgStart][i1-classesStart][j1-imgStart] >= 0) :
                    cur_dist = distances[i-classesStart][j-imgStart][i1-classesStart][j1-imgStart]
                    cur_class = i1
        if (i == cur_class) :
            correct += 1
print(correct)                                  
print(1. * correct / all)  