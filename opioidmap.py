#!/usr/bin/env python3

#remove useless imports
import random
from collections import Counter

import csv
import sys
import numpy
import scipy.spatial
import statistics
import bisect
import json
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import time

# Set up fit and predict to be called on an object just like the sample library he showed us
trainingDataFile = open(sys.argv[1])
trainingCsvReader = csv.reader(trainingDataFile)

dataRep = []
for line in trainingCsvReader:
    dataRep.append(line)


data = []

for i in range(1, len(dataRep)-2, 2):
    line = dataRep[i]
    line.append(dataRep[i+1])
    data.append(line)
    #data = data.astype(numpy.float)
    #classification = line[0]
    #trainDatum = DataObject(classification, data)
    #self.trainingData.append(trainDatum)

print(data[0])
print()
print(data[1])
print()
print(data[2])
print()
