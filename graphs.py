
#remove useless imports
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
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

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
    datum[31] = (coord1, coord2);

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
    classArray.append(datum[17])
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
    # if (datum[3] == "White"):
    #     # datum[3] = (1,0,0,1);
    #     datum[3] = 0
    # elif (datum[3] == "Black"):
    #     # datum[3] = (0,0,1,0);
    #     datum[3] = 1
    # elif (datum[3] == "Asian"):
    #     # datum[3] = (0,1,0,0);
    #     datum[3] = 2
    # elif (datum[3] == "Hispanic"):
    #     # datum[3] = (1,0,0,0);
    #     datum[3] = 3
    # else:
    #     # datum[3] = (0,0,0,0);
    #     datum[3] = 0

    #Sex
    # if (datum[2] == "Male"):
    #     datum[2] = (0,1);
    # elif (datum[2] == "Female"):
    #     datum[2] = (1,0);
    # else:
    #     datum[2] = (0,0);

    #Date
    overallDate = 0
    if (datum[1] == ''):
        continue
    datum[1] = datum[1].replace('/', '')
    date = datetime.datetime.strptime(datum[1], "%m%d%Y").date()
    year = date.year
    month = date.month
    day = date.day

    overallDate += (year-2012) * 360
    overallDate += month * 30 #TODO Fix for months w/ strange dates
    overallDate += day

    datum[1] = overallDate

    #CaseNumber
    del datum[0]

race =[]
for d in dataArray:
    race.append(d[2])

print(dataArray)

dates =[]

for d in dataArray:
    dates.append(d[0])

print(dates)

for i in range(len(race)):
    if race[i] == '':
        race[i] = 'Unknown'

f = pd.value_counts(race)

raceCount = []
for j in f:
    raceCount.append(j)

y_pos = np.arange(len(raceCount))

races = ['White', 'Hispanic, White', 'Black', 'Unknown', 'Hispanic, Black', 'Asian, Other', 'Asian Indian',
         'Chinese', 'Other']

# Dates vs. Frequency

# plt.hist(dates, bins=10)
sns.distplot(dates, bins=10)
plt.title('Date vs. Frequency')
plt.xlim(0,2000)
plt.xlabel('Date')
plt.ylabel('Frequency')

# Race vs. Frequency

plt.figure(figsize=(12,8))
sns.barplot(y_pos, raceCount, palette='GnBu_d')
# plt.bar(y_pos, raceCount)
plt.title('Race vs. Frequency')
plt.xlabel('Race')
plt.ylabel('Frequency')
plt.yticks(np.linspace(0,900,10))
plt.xticks(y_pos, races)
plt.show()

# Gender vs. Frequency

gender = []
for d in dataArray:
    gender.append(d[1])

g = pd.value_counts(gender)

gCount = [g[0], g[1]]
y_pos2 = np.arange(len(gCount))
genders = ['Male', 'Female']

sns.barplot(y_pos2, gCount)
plt.title('Gender vs. Frequency')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(y_pos2, genders)
plt.show()

