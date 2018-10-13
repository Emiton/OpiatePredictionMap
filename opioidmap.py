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

data = []

for line in trainingCsvReader:
    data.append(line)
    #data = data.astype(numpy.float)
    #classification = line[0]
    #trainDatum = DataObject(classification, data)
    #self.trainingData.append(trainDatum)

print(data[0])
print(data[1])
print(data[2])
