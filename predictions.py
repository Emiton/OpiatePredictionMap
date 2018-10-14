# remove useless imports
import random
from collections import Counter

import csv
import sys
# import numpy
# import scipy.spatial
# import statistics
# import bisect
# import json
# import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


import time
import datetime
import re

# Set up fit and predict to be called on an object just like the sample library he showed us
trainingDataFile = open('data.csv')
trainingCsvReader = csv.reader(trainingDataFile)

dataRep = []
for line in trainingCsvReader:
    dataRep.append(line)

data = []

for i in range(1, len(dataRep) - 2, 2):
    line = dataRep[i]
    # line.append(dataRep[i+1])
    data.append(line)
    # data = data.astype(numpy.float)
    # classification = line[0]
    # trainDatum = DataObject(classification, data)
    # self.trainingData.append(trainDatum)

# print(data[0])
# print("K")
# print(data[1])
# print("K")
# print(data[2])
# print("K")


dataArray = []
for dataPoint in data:
    oneLine = []
    for dim in dataPoint:
        oneLine.append(dim)

    # If the overdose involved heroin
    # Maybe include data for Morphine and Any Opioid and NOT take any of them as givens? TODO
    if ((oneLine[15] == 'Y') or (oneLine[15] == 'y')):
        dataArray.append(oneLine)

# CaseNumber,Date,Sex,Race,Age,Residence City,Residence State,Residence County,Death City,
# Death State,Death County,Location,DescriptionofInjury,InjuryPlace,ImmediateCauseA,
# Heroin,Cocaine,Fentanyl,Oxycodone,Oxymorphone,EtOH,Hydrocodone,Benzodiazepine,Methadone,
# Amphet,Tramad,Morphine (not heroin),Other,Any Opioid,MannerofDeath,AmendedMannerofDeath,DeathLoc
classArray = []
# Parse features in reverse order (preserves list size for deletes)
for datum in dataArray:

    # DeathLoc
    coords = datum[31].split(',')
    coord1 = float(re.sub("[^0123456789\.]", "", coords[1]))
    coord2 = float(re.sub("[^0123456789\.]", "", coords[2]))
    datum[30] = coord1
    datum[31] = coord2

    # Cocaine - AmendeMannerOfDeath
    # del datum[30]
    del datum[29]
    del datum[28]
    del datum[27]
    del datum[26]
    del datum[25]
    del datum[24]
    del datum[23]
    del datum[22]
    del datum[21]
    del datum[20]
    del datum[19]
    del datum[18]
    # Append the fentanyl classification
    if datum[17] == '':
        classArray.append(0)
    else:
        classArray.append(1)
    del datum[17]
    del datum[16]
    # Consider other drugs? Maybe it's unimportant that we don't understand how these drugs would interact with heroin to contribute to an overdose being counted

    # Heroin (given)
    del datum[15]

    # ImmediateCauseA - Resident City
    del datum[14]
    del datum[13]
    del datum[12]
    del datum[11]
    del datum[10]
    del datum[9]
    del datum[8]
    del datum[7]
    del datum[6]
    del datum[5]
    del datum[4]
    del datum[3]
    del datum[2]


    # Date
    overallDate = 0
    if (datum[1] == ''):
        continue
    datum[1] = datum[1].replace('/', '')
    date = datetime.datetime.strptime(datum[1], "%m%d%Y").date()
    year = date.year
    month = date.month
    day = date.day

    overallDate += (year - 2012) * 360
    overallDate += month * 30  # TODO Fix for months w/ strange dates
    overallDate += day

    datum[1] = overallDate

    # CaseNumber
    del datum[0]
    # del datum[4]
    # del datum[4]

# Geographic Normalization

DateC = []
Geo1C = []
Geo2C = []

for d in dataArray:
    DateC.append(d[0])
    Geo1C.append(d[1])
    Geo2C.append(d[2])

DCmin = min(DateC)
DCmax = max(DateC)
G1Cmin = min(Geo1C)
G1Cmax = max(Geo1C)
G2Cmin = min(Geo2C)
G2Cmax = max(Geo2C)

for d in dataArray:
    d[0] = (d[0] - DCmin) / (DCmax - DCmin)
    d[1] = (d[1] - G1Cmin) / (G1Cmax - G1Cmin)
    d[2] = (d[2] - G2Cmin) / (G2Cmax - G2Cmin)

print(dataArray)

X_train, X_test, y_train, y_test = train_test_split(dataArray, classArray, test_size=0.2)
#print(X_train[0])

## Naive Bayes
print('=' * 20, 'Naive Bayes', '=' * 20)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy is ', accuracy_score(y_test, y_pred))

## Linear Regression
# print('=' * 20, 'Linear Regression', '=' * 20)
#
# from sklearn.linear_model import LinearRegression
# classifier = LinearRegression()
# classifier.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)
#
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print('Accuracy is ', accuracy_score(y_test, y_pred))

## Logistic Regression
print('=' * 20, 'Logistic Regression', '=' * 20)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy is ', accuracy_score(y_test, y_pred))

## Support Vector Machine
print('=' * 20, 'Support Vector Machine', '=' * 20)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy is ', accuracy_score(y_test, y_pred))

## Decision Trees
print('=' * 20, 'Decision Trees', '=' * 20)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy is ', accuracy_score(y_test, y_pred))

## k Nearest Neighbors
print('=' * 20, 'k Nearest Neighbors', '=' * 20)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy is ', accuracy_score(y_test, y_pred))