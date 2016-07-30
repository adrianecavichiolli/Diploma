import csv
import pandas as pd
import scipy.spatial.distance as dist
import sklearn.metrics.pairwise as prw
from skimage.io import imread
import numpy as np
import math
from sklearn.metrics import roc_curve 
from matplotlib import pyplot

#format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
def testHist(format, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
    #cnts = dataset['Cnt']
    fnList = dataset['path']
    bins = [16, 32, 64,128,256]
    out = open(output, 'w')

    for bin in bins :
        histograms = list(map(lambda x:  np.histogram(imread(format.format(x)), bins=range(bin))[0], fnList))
        distances = prw.pairwise_distances(histograms, metric='l1')
        np.fill_diagonal(distances, math.inf)

        guessedClasses = np.apply_along_axis(lambda x: np.argmin(x), 1, distances)
        correct = list(map(lambda i: idxCls[guessedClasses[i]] == idxCls[i], range(0, np.alen(idxCls))))
        maxDist = np.max(distances)
        scores = np.apply_along_axis(lambda x: 1. * np.min(x) / maxDist, 1, distances)
        fpr, tpr, thresholds = roc_curve(correct, scores, pos_label=1)
        
        out.write(str(np.average(correct)))
        print(str(np.average(correct)))
        pyplot.plot(tpr, fpr)
     #   pyplot.show()
