import csv
import pandas as pd
import scipy.spatial.distance as dist
import sklearn.metrics.pairwise as prw
from skimage.io import imread
import numpy as np
import math
from skimage.feature import local_binary_pattern
from sklearn.metrics import roc_curve 
from matplotlib import pyplot
import cv2

lbpP = 8
lbpR = 1
lbpMethod = 'default'

#format = 'F:/studies/diploma/DataSets-Computer-Vision/data.brodatz/size_213x213/{}'

#dataset = pd.read_csv('F:/studies/diploma/DataSets-Computer-Vision/data.brodatz/size_213x213/index.csv')
def testLBP (format, formatMask, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
   # cnts = dataset['Cnt']
    fnList = dataset['path']
  #  out = open(output, 'w')
    lbps = list(map(lambda x: local_binary_pattern(cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), lbpP, lbpR, lbpMethod), fnList))
    histograms = list(map(lambda x:  np.histogram(x, bins=range(int(np.max(lbps)) + 1))[0], lbps))
    distances = prw.pairwise_distances(histograms, metric='l1')
    np.fill_diagonal(distances, math.inf)
    guessedClasses = np.apply_along_axis(lambda x: np.argmin(x), 1, distances)
    scores = np.apply_along_axis(lambda x: np.min(x), 1, distances)
    correct = list(map(lambda i: idxCls[guessedClasses[i]] == idxCls[i], range(0, np.alen(idxCls))))
   # out.write(str(np.average(correct)))
  #  fpr, tpr, thresholds = roc_curve(correct, scores, pos_label=1)
  #  pyplot.plot(tpr, fpr)
   # pyplot.show()
    with open(output + 'lbp_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(distances)

    with open(output + 'lbp_guessedClasses.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(guessedClasses)

    with open(output + 'lbp_correct.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(correct)

    with open(output + 'lbp_real.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(idxCls)