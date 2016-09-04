import COOMatrixSpeedUp
import HistogramDescriptorSpeedUp
#import CorrelationMatcher
#import ImageMatching
from matplotlib import pyplot as plt
import cv2
import numpy as np

import matplotlib.image as mpimg
import skimage.exposure as exposure
import math
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import skimage.io as io
import cv2
import pandas as pd
from skimage.io import imread
from scipy.stats import pearsonr
import csv

#format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
#formatMask = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}_proc_mask.png'
#path = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx1.csv'

def testCorrelation (format, formatMask, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
    fnList = dataset['path']
    masks = list(map(lambda x: imread(formatMask.format(x)), fnList))
    #im = masks[0]

    #white = np.nonzero(im)
    #left = min(white[0])
    #right = max(white[0])
    #top = min(white[1])
    #bottom = max(white[1])

    for mask in masks :
        if np.sum(mask) == 0:
            mask[len(mask) - 1][len(mask[0]) - 1] = 1

    left = list(map(lambda x: min(np.nonzero(x)[1]), masks))
    right = list(map(lambda x: max(np.nonzero(x)[1]), masks))
    top =  list(map(lambda x: min(np.nonzero(x)[0]), masks))
    bottom = list(map(lambda x: max(np.nonzero(x)[0]), masks))

    images = list(map(lambda x:  imread(format.format(x)), fnList))
    images_croped = list(map(lambda x:  imread(format.format(x)), fnList))
    for i in range(0, len(images) - 1):
        images_croped[i] = images[i][top[i] : bottom[i], left[i] : right[i]] #Y[[0,3],:][:,[0,3]]
        #if len(images[i]) > 0 and len(images[i][0]) > 0 :
        #    images_croped[i] = np.pad(images[i], ((0,512 - len(images[i])),(0, 512 - len(images[i][0]))), mode='constant', constant_values=0)
 #   correlations = []
    max_correlations = np.empty([len(images), len(images)])
    for i in range(0, len(images)):
        for j in range(0, len(images)):
            if i == j :
              #  correlations[i][j].append(0.)
                max_correlations[i][j] = 0.
            elif len(images_croped[j]) > 0 and len(images_croped[j][0]) > 0 : 
                max_correlations[i][j] = np.max(cv2.matchTemplate(images[i], images_croped[j], cv2.TM_CCORR_NORMED))
                #correlations.append(cv2.matchTemplate(images[i], images_croped[j], cv2.TM_CCORR_NORMED))

    with open(output + 'correlation_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(max_correlations)

    #with open(output + 'correlation_matrixes.csv', 'w', newline='') as fp:
    #    a = csv.writer(fp, delimiter=',')
    #    a.writerows(correlations)

    with open(output + 'correlation_real.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(idxCls)


#print(cv2.__version__)
#sift = cv2.xfeatures2d.SIFT_create()
format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
path = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx1.csv'
pathSubset4x100 = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idxSubset_4_100.csv'
formatMask = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}_proc_mask.png'
#thresholdSft = .7
#thresholdSrf = .7

#HistogramDescriptorSpeedUp.testHist(format, formatMask, pathSubset4x100, 'MaskedSubset')
#COOMatrixSpeedUp.testCOO(format, formatMask, pathSubset4x100, 'MaskedSubset')
#LBPSpeedUp.testLBP(format, formatMask, pathSubset4x100, 'MaskedSubset')

#HistogramDescriptorSpeedUp.testHist(format, formatMask, path, 'Maskedset')
#COOMatrixSpeedUp.testCOO(format, formatMask, path, 'Maskedset')
#LBPSpeedUp.testLBP(format, formatMask, path, 'Maskedset')
#lstLegend=['sift', 'surf']
#plt.figure()
#plt.hold(True)
#true_positive = np.empty(19)
#false_positive = np.empty(19)
#for step in range (1, 20) :
#    tr = .01 * step + .8
#    ind = step - 1

#    false_positive[ind], true_positive[ind] = ImageMatching.testMatch(format, formatMask, path, 'matchSift.txt', 'sift', tr)
    #fprSu, tprSu = ImageMatching.testMatch(format, formatMask, path, 'matchSurf.txt', 'surf', tr)

#plt.hold(False)
#plt.grid(True)
#plt.plot(false_positive, true_positive)
#plt.show()
testCorrelation(format, formatMask, pathSubset4x100, 'SubsetCorrelation')
testCorrelation(format, formatMask, path, 'SetCorrelation')