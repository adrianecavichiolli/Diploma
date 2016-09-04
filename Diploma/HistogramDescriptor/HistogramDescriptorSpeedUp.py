import csv
import pandas as pd
import scipy.spatial.distance as dist
import sklearn.metrics.pairwise as prw
from skimage.io import imread
import numpy as np
import math
from sklearn.metrics import roc_curve 
from matplotlib import pyplot
import cv2

#format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
def testHist(format, formatMask, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
    #cnts = dataset['Cnt']
    fnList = dataset['path']
    #out = open(output, 'w')

    histograms = list(map(lambda x:  np.histogram(cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), bins=range(16))[0], fnList))
    distances = prw.pairwise_distances(histograms, metric='l1')
    np.fill_diagonal(distances, math.inf)

    guessedClasses = np.apply_along_axis(lambda x: np.argmin(x), 1, distances)
    correct = list(map(lambda i: idxCls[guessedClasses[i]] == idxCls[i], range(0, np.alen(idxCls))))
    maxDist = np.max(distances)
    scores = np.apply_along_axis(lambda x: 1. * np.min(x) / maxDist, 1, distances)
      #  fpr, tpr, thresholds = roc_curve(correct, scores, pos_label=1)
        
     #   out.write(str(np.average(correct)))
     #   print(str(np.average(correct)))
     #   pyplot.plot(tpr, fpr)
     #   pyplot.show()
    with open(output + 'histogram_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(distances)

    with open(output + 'histogram_guessedClasses.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(guessedClasses)

    with open(output + 'histogram_correct.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(correct)

    with open(output + 'histogram_real.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(idxCls)