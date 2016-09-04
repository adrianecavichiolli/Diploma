from __future__ import print_function

import numpy as np
import cv2
import pandas as pd
from skimage.io import imread
from sklearn.metrics import roc_curve 
from matplotlib import pyplot

import csv

#from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

#detector = cv2.BRISK_create()
#flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                                   table_number = 6, # 12
#                                   key_size = 12,     # 20
#                                   multi_probe_level = 1) #2
#matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def match(kp1, kp2, desc1, desc2, matcher):
    if desc1 is None or desc2 is None :
        return 0.
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, 0.
    if status is None:
        status = 0.
    return np.mean(status)
            #vis = explore_match(win, img1, img2, kp_pairs, status, H)

def testMatch(format, formatMask, path, output, name) :
    detector, matcher = init_feature(name)
   # format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'

    dataset = pd.read_csv(path)
    idxCls = dataset['idx']
    #cnts = dataset['Cnt']
    fnList = dataset['path']
    #images = list(map(lambda x:  imread(format.format(x)), fnList))
    images = list(map(lambda x: cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), fnList))

    kpsAndDescriptors = list(map(lambda x: detector.detectAndCompute(x, None), images))

    kps = list(map(lambda x: x[0], kpsAndDescriptors))
    descriptors = list(map(lambda x: x[1], kpsAndDescriptors))
    distances = np.empty((len(descriptors), len(descriptors)))

    for i in range(0, len(descriptors)) :
        for j in range(0, len(descriptors)) :
            if (i == j) :
                distances[i][j] = 1.1
                continue
            distances[i][j] = 1. - match(kps[i], kps[j], descriptors[i], descriptors[j], matcher);

    #tpr = 0.
    #fpr = 0.

    guessedClasses = np.apply_along_axis(lambda x: np.argmin(x), 1, distances)

  #  for i in range(0, len(descriptors)) :
  #      k = np.argmin(distances[i])
  #      #scores[i] = 
  #      if 1. - distances[i][k] >= threshold:
  #          if idxCls[i] == idxCls[k] :
  #              tpr = tpr + 1.0
  #          else:
  #              fpr = fpr + 1.0
  #  fpr, tpr, thresholds = roc_curve(guessed, scores, pos_label=1)
  #  print(fpr)
 #   print(tpr)
 #   print(thresholds)
   # pyplot.plot(tpr, fpr)
 #   pyplot.show()
    
    #out = '{}.csv'.format(output)
    #with open(out, 'wb') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=' ',
    #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #    writer.writerows(distances)
    with open(output + 'matching_descriptors.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(descriptors)

    with open(output + 'matching_distances.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(distances)

    with open(output + 'matching_guessedClasses.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(guessedClasses)

    with open(output + 'matching_correct.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(correct)

    #fpr = fpr / len(descriptors)
    #tpr = tpr / len(descriptors)
    #return fpr, tpr

format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
path = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx1.csv'
pathSubset4x100 = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idxSubset_4_100.csv'
formatMask = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}_proc_mask.png'

testMatch(format, formatMask, pathSubset4x100, "Subset", "sift")
testMatch(format, formatMask, pathSubset4x100, "Set", "sift")