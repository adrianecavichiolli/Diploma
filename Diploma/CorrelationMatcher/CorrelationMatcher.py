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
import skimage.io as io
import cv2
import pandas as pd
from skimage.io import imread
from scipy.stats import pearsonr

#format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
#formatMask = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}_proc_mask.png'
#path = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx1.csv'

def testCorrelation (format, formatMask, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
    fnList = dataset['path']
    masks = list(map(lambda x: imread(formatMask.format(x)), fnList))
    im = masks[0]

    white = np.nonzero(im)
    left = min(white[0])
    right = max(white[0])
    top = min(white[1])
    bottom = max(white[1])

    for mask in masks :
        if np.sum(mask) == 0:
            mask[len(mask) - 1][len(mask[0]) - 1] = 1

    left = list(map(lambda x: min(np.nonzero(x)[1]), masks))
    right = list(map(lambda x: max(np.nonzero(x)[1]), masks))
    top =  list(map(lambda x: min(np.nonzero(x)[0]), masks))
    bottom = list(map(lambda x: max(np.nonzero(x)[0]), masks))

    images = list(map(lambda x:  imread(format.format(x)), fnList))
    images_croped = np.empty(len(images))
    for i in range(0, len(images) - 1):
        images_croped[i] = images[i][top[i] : bottom[i], left[i] : right[i]] #Y[[0,3],:][:,[0,3]]
        if len(images[i]) > 0 and len(images[i][0]) > 0 :
            images_croped[i] = np.pad(images[i], ((0,512 - len(images[i])),(0, 512 - len(images[i][0]))), mode='constant', constant_values=0)
    correlations = np.empty([len(images), len(images)])
    max_correlations = np.empty([len(images), len(images)])
    for i in range(0, len(images)):
        for j in range(0, len(images)):
            if i == j :
                correlations[i][j] = 0.
            elif len(images[i]) > 0 and len(images[j]) > 0 : 
                correlations[i][j] = np.max(cv2.matchTemplate(images[i], images_croped[j], cv2.TM_CCORR_NORMED))
                max_correlations[i][j] = cv2.matchTemplate(images[i], images_croped[j], cv2.TM_CCORR_NORMED)

    with open(output + 'correlation_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(max_correlations)

    with open(output + 'correlation_matrixes.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(correlations)

    with open(output + 'correlation_real.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(idxCls)
    #true_positive = np.empty(19)
    #false_positive = np.empty(19)
    #for step in range (1, 20) :
    #    threshold = .01 * step + .8
    #    ind = step - 1
    #    true_positive[ind] = 0.
    #    false_positive[ind] = 0.
    #    scores = np.zeros(len(correlations))
    #    for i in range(0, len(correlations)) :
    #        k = np.argmax(correlations[i])
    #        scores[i] = correlations[i][k]
    #        if idxCls[i] == idxCls[k] and scores[i] >= threshold :
    #            true_positive[ind] = true_positive[ind] + 1. 
    #        elif idxCls[i] != idxCls[k] and scores[i] >= threshold:
    #            false_positive[ind] = false_positive[ind] + 1. 
    #    true_positive[ind] = true_positive[ind] / len(images)
    #    false_positive[ind] = false_positive[ind] / len(images)
    #    print (true_positive)
    #    print(false_positive)
    #plt.figure()
    #plt.plot(false_positive, true_positive)
    #plt.show()
print ('Hello world')