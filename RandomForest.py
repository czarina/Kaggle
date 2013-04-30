#Random Forest approach to the Titanic problem: use the
#scikit-learn package to contruct an ensemble of decision
#trees with random samplings of training data; predict 
#survival of Titanic passengers by outputting mode of 
#classifiers produced by the individual trees 

import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

np.set_printoptions(threshold=np.nan)   

def cleanAndPrep(data):
	#Prepare data for random trees
	#Convert data into floats when possible; remove extraneous rows,
	#and fill in data where possible

	#Remove name, cabin, ticket
	
	data = [(float(x[0]), float(x[1]), x[3], x[4], float(x[5]), float(x[6]), x[8], x[10]) for x in data]
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
	data[data[0::,7]=="", 7]=1.0

	#Fill in fare
	medianFareList = []
	for x in data[0::,6]:
		if(x>0 and x!=''):
			medianFareList.append(float(x))
	data[data[0::,6]=='0.0', 6]=float(np.median(medianFareList))
	data[data[0::,6]=='', 6]=float(np.median(medianFareList))

	return data.astype(float)

#Open up training data, csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = csv_file_object.next()
train_data=[]                          
for row in csv_file_object:      
    train_data.append(row)           
train_data = np.array(train_data)  

#Clean and Prepare training data
train_data = cleanAndPrep(train_data)

#Open up test data, csv file in to a Python object

test_file_object = csv.reader(open('test.csv', 'rU'))
header = test_file_object.next()

test_data=[]                          
for row in test_file_object:      
    row.insert(0,0)
    test_data.append(row)             
test_data = np.array(test_data)  

#Clean and Prepare test data
test_data = cleanAndPrep(test_data)
test_data = test_data[0::, 1:8]

Forest = RandomForestClassifier(n_estimators = 100)      
Forest = Forest.fit(train_data[0::,1::], train_data[0::,0]) 
Output = Forest.predict(test_data)                       

#Prepare output
test_file_object = csv.reader(open('test.csv', 'rU'))
header = test_file_object.next()

#Run RandomForest
header.insert(0, "survived")
open_file_object = csv.writer(open("RandomForest.csv", "wb"))
open_file_object.writerow(header)
count = 0

for row in test_file_object:
	if(Output[count]>0):
		row.insert(0, 1)
	else:
		row.insert(0, 0)
	open_file_object.writerow(row)
	count = count+1


