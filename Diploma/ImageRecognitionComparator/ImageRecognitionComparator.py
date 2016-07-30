#import COOMatrixSpeedUp
#import HistogramDescriptorSpeedUp
#import LBPSpeedUp
import ImageMatching
from matplotlib import pyplot as plt
import cv2
#print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()
format = 'F:\studies\diploma\Diploma\Data-XRay-Resized\8\{}'
path = 'F:\studies\diploma\Diploma\Data-XRay-Resized\idx.csv'
#HistogramDescriptorSpeedUp.testHist(format, path, 'hist.txt')
#COOMatrixSpeedUp.testCOO(format, path, 'coomatrix.txt')
#LBPSpeedUp.testLBP(format, path, 'lbp.txt')
lstLegend=['sift', 'surf']
plt.figure()
plt.hold(True)

fprSf, tprSf, thresholdsSf = ImageMatching.testMatch(format, path, 'match.txt', 'sift')
fprSu, tprSu, thresholdsSu = ImageMatching.testMatch(format, path, 'match.txt', 'surf')
plt.plot(tprSf, fprSf)
plt.plot(tprSu, fprSu)
plt.hold(False)
plt.grid(True)
plt.legend(lstLegend)
plt.show()
