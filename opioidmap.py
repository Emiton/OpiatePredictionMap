#!/usr/bin/env python3

#remove useless imports
import random
from collections import Counter

import csv
import sys
import numpy
import scipy.spatial
# import statistics
# import bisect
# import json
# import math
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

import time
import datetime
import re
from sklearn.linear_model import LinearRegression


# Set up fit and predict to be called on an object just like the sample library he showed us
trainingDataFile = open(sys.argv[1])
trainingCsvReader = csv.reader(trainingDataFile)

dataRep = []
for line in trainingCsvReader:
    dataRep.append(line)


data = []

for i in range(1, len(dataRep)-2, 2):
    line = dataRep[i]
    #line.append(dataRep[i+1])
    data.append(line)
    #data = data.astype(numpy.float)
    #classification = line[0]
    #trainDatum = DataObject(classification, data)
    #self.trainingData.append(trainDatum)

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
    #Maybe include data for Morphine and Any Opioid and NOT take any of them as givens? TODO
    if ((oneLine[15] == 'Y') or (oneLine[15] == 'y')):
        dataArray.append(oneLine)

#CaseNumber,Date,Sex,Race,Age,Residence City,Residence State,Residence County,Death City,
#Death State,Death County,Location,DescriptionofInjury,InjuryPlace,ImmediateCauseA,
#Heroin,Cocaine,Fentanyl,Oxycodone,Oxymorphone,EtOH,Hydrocodone,Benzodiazepine,Methadone,
#Amphet,Tramad,Morphine (not heroin),Other,Any Opioid,MannerofDeath,AmendedMannerofDeath,DeathLoc
classArray = []
#Parse features in reverse order (preserves list size for deletes)
for datum in dataArray:

    
    #DeathLoc
    coords = datum[31].split(',')
    coord1 = re.sub("[^0123456789\.]","",coords[1])
    coord2 = re.sub("[^0123456789\.]","",coords[2])
    del datum[31]
    datum.append(float(coord1))
    datum.append(float(coord2));

    #Cocaine - AmendeMannerOfDeath
    del datum[30]
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
    #Append the fentanyl classification
    if ((datum[17] == 'Y') or (datum[17] == 'y')):
        classArray.append(1.0)
    else:
        classArray.append(0.0)
    del datum[17]
    del datum[16]
    #Consider other drugs? Maybe it's unimportant that we don't understand how these drugs would interact with heroin to contribute to an overdose being counted

    #Heroin (given)
    del datum[15]

    #ImmediateCauseA - Resident City 
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

    #Age
    datum[4] = int(datum[4])

    #Race
    if (datum[3] == "White"):
        del datum[3]
        datum[3:3] = [0,0,0,1]
    elif (datum[3] == "Black"):
        del datum[3]
        datum[3:3] = [0,0,1,0]
    elif (datum[3] == "Asian"):
        del datum[3]
        datum[3:3] = [0,1,0,0]
    elif (datum[3] == "Hispanic"):
        del datum[3]
        datum[3:3] = [1,0,0,0]
    else:
        del datum[3]
        datum[3:3] = [0,0,0,0]

    #Sex
    if (datum[2] == "Male"):
        del datum[2]
        datum[2:2] = [0,1]
    elif (datum[2] == "Female"):
        del datum[2]
        datum[2:2] = [1,0]
    else:
        del datum[2]
        datum[2:2] = [0,0]

    #Date
    overallDate = 0
    if (datum[1] == ''):
        datum[1] = (2017-2012) * 365
    datum[1] = datum[1].replace('/', '')
    date = datetime.datetime.strptime(datum[1], "%m%d%Y").date()
    year = date.year
    month = date.month
    day = date.day

    overallDate += (year-2012) * 365
    overallDate += month * 30 #TODO Fix for months w/ strange dates
    overallDate += day

    datum[1] = overallDate

    #CaseNumber
    del datum[0]

for line in dataArray:
    if (len(line) != len(dataArray[0])):
        print("DISTANCE ISSUE")

print()
print(dataArray[1])
print()
print(dataArray[2])

#dataArray = numpy.reshape(dataArray, len(dataArray), len(dataArray[0]))
for arr in dataArray:
    arr = numpy.array(arr)
dataArray = numpy.array(dataArray)
#print(dataArray)
classArray = numpy.array(classArray)
print(dataArray)
print(classArray)
reg = LinearRegression().fit(dataArray, classArray)
print(reg.predict(numpy.array([[1024, 0, 1, 0, 0, 0, 1, 67, 41.308252, 72.924161]])))
#print(len(dataArray))
