import numpy as np

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.exposure as exposure
import math
format = 'F:/studies/diploma/DataSets-Computer-Vision/data.brodatz/size_28x28/D{:03d}_{:02d}.png'
classesStart = 1
classesEnd = 2

imgStart = 1
imgEnd = 3
#imgLearnStart = 6
#imgLearnEnd = 10
#imgClassstart = 1
#imgClassEnd = 6

def classify (numberCl, numberInst, L) :
    imgName = format.format(numberCl, numberInst)
    img=cv2.imread(imgName)
    if (img is None) :
        return 0
    curClass = -1
    minres = 1000000.0
    for i in range (classesStart, classesEnd):
        for j in range (imgStart, imgEnd):
            if (i == numberCl and j == numberInst) :
                continue
            imgCurName = format.format(i, j)
            imgCur = cv2.imread(imgCurName)
            if (imgCur is None) :
                continue
            curres = L(img, imgCur)
            if (curres < minres) :
                minres = curres
                curClass = i
    return curClass



def imageL1(img1, img2 ):
   hist1 = np.histogram(img1, bins=256)[0]
   hist2 = np.histogram(img2, bins=256)[0]
   plt.hold(True)
   plt.plot(hist1)
   plt.plot(hist2)
   plt.hold(False)
   plt.show()
   result = abs(hist1 - hist2)
   return np.sum(result)

def imageL2(img1, img2 ):
   hist1 = np.histogram(img1, bins=256)[0]
   hist2 = np.histogram(img2, bins=256)[0]
  
   result = (hist1 - hist2)**2
   return math.sqrt(np.sum(result))

def testL (L) :
    correct = 0
    all = (imgEnd - imgStart) * (classesEnd - classesStart)
    for i in range (classesStart, classesEnd):
        for j in range (imgStart, imgEnd):
            fndClass = classify(i, j, L)
            if (fndClass == 0) :
                all -= 1
                continue
            if (i == fndClass) :
                correct += 1
    print(correct)
    print(1. * correct / all)

testL(imageL1)
testL(imageL2)
