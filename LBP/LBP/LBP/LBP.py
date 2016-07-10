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
format = 'F:/studies/diploma/DataSets-Computer-Vision/data.brodatz/size_28x28/D{:03d}_{:02d}.png'
classesStart = 1
classesEnd = 112

imgStart = 1
imgEnd = 10

lbpP = 8
lbpR = 1
lbpMethod = 'default'
def calculate_lbp(image) :
    lbp = local_binary_pattern(image, lbpP, lbpR, lbpMethod)
    return lbp

def get_image(cls, img) :
    imgName = format.format(cls, img)
    image=imread(imgName)
    return image


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
                    result[i-clS][j-imgS][i1-clS][j1-imgS] = imageLBPL1(get_image(i, j), get_image(i1, j1))
    return result

def imageLBPL1(img1, img2) :
    lbp1 = calculate_lbp(img1)
    lbp2 = calculate_lbp(img2)
    hist1 = np.histogram(lbp1, bins=64)[0]
    hist2 = np.histogram(lbp2, bins=64)[0]
    result = abs(hist1 - hist2)
    return np.sum(result)

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