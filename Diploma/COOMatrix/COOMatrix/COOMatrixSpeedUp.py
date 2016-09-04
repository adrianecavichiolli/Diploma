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

offset = 5
from skimage.feature import greycomatrix

def testCOO (format, formatMask, path, output) :
    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
#cnts = dataset['Cnt']
    fnList = dataset['path']
    bins = [16, 32, 64,128,256]
    out = open(output, 'w')

  #  for bin in bins :
    histograms = list(map(lambda x:  np.ravel(greycomatrix(cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), [offset], [0, math.pi / 4., math.pi / 2., 3. * math.pi / 4.], 256, symmetric=True, normed=True)), fnList))
    distances = prw.pairwise_distances(histograms, metric='l1')
    np.fill_diagonal(distances, math.inf)

    guessedClasses = np.apply_along_axis(lambda x: np.argmin(x), 1, distances)
    scores = np.apply_along_axis(lambda x: np.min(x), 1, distances)
    correct = list(map(lambda i: idxCls[guessedClasses[i]] == idxCls[i], range(0, np.alen(idxCls))))
       # print(np.average(correct))
 #   out.write(str(np.average(correct)))
  #  fpr, tpr, thresholds = roc_curve(correct, scores, pos_label=1)
      #  print(fpr)
      #  print(tpr)
      #  print(thresholds)
 #   pyplot.plot(tpr, fpr)
    #    pyplot.show()
    with open( output + 'coo_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(distances)

    with open(output + 'coo_guessedClasses.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(guessedClasses)

    with open(output + 'coo_correct.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(correct)
   
    with open(output + 'coo_real.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(idxCls)