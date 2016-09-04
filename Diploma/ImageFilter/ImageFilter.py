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

format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
formatMask = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}_proc_mask.png'
pathLearn = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idxLearn.csv'
pathTest = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx2.csv'

threshold_hor = .85
threshold_vert = .85
k = 1
#imgName = 'F:/studies/diploma/Diploma/Data-XRay-Resized/1.png'
#image=io.imread(imgName)
#plt.figure()
#plt.imshow(image)
##plt.show()
#vertical_profile = np.sum(image, axis=0)
#horizontal_profile = np.sum(image, axis=1)
#plt.figure()
#plt.plot(range(0, np.alen(vertical_profile)), vertical_profile)
#plt.figure()
#plt.plot(range(0, np.alen(horizontal_profile)), horizontal_profile)
#plt.show()
datasetLearn = pd.read_csv(pathLearn)
idxClsLearn = datasetLearn['idx']
fnListLearn = datasetLearn['path']
imagesLearn = list(map(lambda x:  imread(format.format(x)), fnListLearn))
imagesMaskedLearn = list(map(lambda x: cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), fnListLearn))
#for img in imagesMaskedLearn :
#    plt.figure()
#    plt.imshow(img)
#    plt.show()

vertical_profile_learn = list(map(lambda x: np.sum(x, axis=0), imagesMaskedLearn))
horizontal_profile_learn = list(map(lambda x: np.sum(x, axis=1), imagesMaskedLearn))

datasetTest = pd.read_csv(pathTest)
idxClsTest = datasetTest['idx']
fnListTest = datasetTest['path']
imagesTest = list(map(lambda x:  imread(format.format(x)), fnListTest))
imagesMaskedTest = list(map(lambda x: cv2.bitwise_and(imread(format.format(x)),imread(formatMask.format(x))), fnListTest))


vertical_profile_test = list(map(lambda x: np.sum(x, axis=0), imagesMaskedTest))
horizontal_profile_test = list(map(lambda x: np.sum(x, axis=1), imagesMaskedTest))

correlations_hor = np.empty((len(horizontal_profile_test), len(horizontal_profile_learn)))
correlations_vert = np.empty((len(horizontal_profile_test), len(horizontal_profile_learn)))
for i in range(0, len(horizontal_profile_test)):
    for j in range(0, len(horizontal_profile_learn)):
        correlations_hor[i][j] = pearsonr(horizontal_profile_test[i], horizontal_profile_learn[j])[0]
        correlations_vert[i][j] = pearsonr(vertical_profile_test[i], vertical_profile_learn[j])[0]

max_correlations_hor = np.empty(len(horizontal_profile_test))
max_correlations_vert = np.empty(len(horizontal_profile_test))
for i in range (0, len(horizontal_profile_test)) :
    max_correlations_hor[i] = np.nanmax(correlations_hor[i])
    max_correlations_vert[i] = np.nanmax(correlations_vert[i])
    if (max_correlations_hor[i] < threshold_hor and max_correlations_vert[i] < threshold_vert) :
        plt.figure()
        plt.imshow(imagesTest[i])
plt.show()

