import csv
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot

def visualize_results(real, correct, distances, guessedClasses, name) :
  #  res = np.mean(correct)
    num_steps = 100
    true_positives = np.empty(num_steps + 1)
    false_positives = np.empty(num_steps + 1)
    for step in np.arange(0, num_steps + 1) :
        threshold = .01 * step
        true_positives[step] = 0.
        false_positives[step] = 0.
        all_positives = 0.
        all_negatives = 0.
        
        for i in range(0, len(guessedClasses)) :
            for k in range(0, len(guessedClasses)) :
                if real[i] == real[k]:
                    all_positives = all_positives + 1.
                else:
                    all_negatives = all_negatives + 1.
                if distances[i][k] >= threshold:
                    if real[i] == real[k] :
                        true_positives[step] = true_positives[step] + 1.0
                    else:
                        false_positives[step] = false_positives[step] + 1.0
        true_positives[step] = true_positives[step] / all_positives
        false_positives[step] = false_positives[step] / all_negatives
    with open('results.txt', 'a', newline='') as fp:
        fp.write('Algorithm: ' + name + '\n')
        fp.write('Number of images: ' + str(len(correct)) + '\n')
        fp.write('Result: ' + str(res)  + '\n')

    pyplot.figure()
    pyplot.plot(true_positives, false_positives)
    pyplot.title(name)
  #  pyplot.show()
    pyplot.savefig(name + '_roc.png')

path_format = 'F:\\studies\\diploma\\Diploma\\result-csvs\\{0}.csv'
names = ['Correlation']
#names = ['Subsetcoo', 'Subsethistogram', 'Subsetlbp', 'setcoo', 'sethistogram', 'setlbp']

for name in names:
    correct = pd.read_csv(path_format.format(name + '_correct'), sep=',',header=None).values[0]

    distances = pd.read_csv(path_format.format(name + '_distances'), sep=',',header=None).values
    np.fill_diagonal(distances, 0.)
    mx = np.max(distances)
    distances = distances / mx
    np.fill_diagonal(distances, mx)
    guessedClasses = pd.read_csv(path_format.format(name + '_guessedClasses'), sep=',',header=None).values[0]
    real = pd.read_csv(path_format.format(name + '_real'), sep=',',header=None).values[0]

    visualize_results(real, correct, distances, guessedClasses, name)


