#Random Forest approach to the Titanic problem: use the
#scikit-learn package to contruct an ensemble of decision
#trees with random samplings of training data; predict 
#survival of Titanic passengers by outputting mode of 
#classifiers output by the individual trees 

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
#Convert data into floats when possible; otherwise remove

#Remove name
data = [(x[0], x[1], x[3], x[4], x[5], x[6], x[8], x[10]) for x in data]
data = np.array(data)

#Convert gender into 0,1
data[data[0::,2]=="male", 2] = 0.0
data[data[0::,2]=="female", 2] = 1.0

#Convert embarkment to 0, 1, 2
data[data[0::,7]=="C", 7] = 0.0
data[data[0::,7]=="Q", 7] = 1.0
data[data[0::,7]=="S", 7] = 2.0

print data