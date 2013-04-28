#Random Forest approach to the Titanic problem: use the
#scikit-learn package to contruct an ensemble of decision
#trees with random samplings of training data; predict 
#survival of Titanic passengers by outputting mode of 
#classifiers produced by the individual trees 

import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

#Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = csv_file_object.next()

data=[]                          
for row in csv_file_object:      
    data.append(row)             
data = np.array(data)  
np.set_printoptions(threshold=np.nan)   

#Prepare data for random trees
#Convert data into floats when possible; remove extraneous rows,
#and fill in data where possible

#Remove name, cabin, ticket
data = [(float(x[0]), float(x[1]), x[3], x[4], float(x[5]), float(x[6]), float(x[8]), x[10]) for x in data]
data = np.array(data)

#Convert gender into 0,1
data[data[0::,2]=="male", 2] = 0.0
data[data[0::,2]=="female", 2] = 1.0

#Fill in and reformat Age
medianList = []
for x in data[0::,3]:
	if(x.isdigit()):
		medianList.append(float(x))
data[data[0::,3]=='', 3]=float(np.median(medianList))
data[0::,3]=[float(x) for x in data[0::,3]]

#Convert embarkment to 0, 1, 2
data[data[0::,7]=="C", 7] = 0.0
data[data[0::,7]=="Q", 7] = 1.0
data[data[0::,7]=="S", 7] = 2.0

#Fill in fare
medianFareList = []
for x in data[0::,6]:
	if(x>0):
		medianFareList.append(float(x))
data[data[0::,6]=='0.0', 6]=float(np.median(medianFareList))

print data

print np.corrcoef(data[0::,1], data[0::,6])
