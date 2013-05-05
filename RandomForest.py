#Random Forest approach to the Titanic problem: use the
#scikit-learn package to contruct an ensemble of decision
#trees with random samplings of training data; predict 
#survival of Titanic passengers by outputting mode of 
#classifiers produced by the individual trees 

import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

np.set_printoptions(threshold=np.nan)   

def cleanAndPrep(data, ifAge):
	#Prepare data for random trees
	#Convert data into floats when possible; remove extraneous rows,
	#and fill in data where possible

	#Remove name, cabin, ticket
	if(ifAge == 1):
		data = [(float(x[0]), float(x[1]), x[3], x[4], float(x[5]), float(x[6]), x[8], x[10]) for x in data]
	else:
		data = [(float(x[0]), float(x[1]), x[3], x[4], float(x[5]), float(x[7]), x[9]) for x in data]

	data = np.array(data)

	#Convert gender into 0,1
	data[data[0::,2]=="male", 2] = 0.0
	data[data[0::,2]=="female", 2] = 1.0

	#Fill in and reformat Age
	if(ifAge==1):
		medianList = []
		for x in data[0::,3]:
			if(x.isdigit()):
				medianList.append(float(x))
		data[data[0::,3]=='', 3]=float(np.median(medianList))
		data[0::,3]=[float(x) for x in data[0::,3]]

	#Convert embarkment to 0, 1, 2
	if(ifAge == 1):
		embarkmentRow = 7
	else:
		embarkmentRow = 6
	data[data[0::,embarkmentRow]=="C", embarkmentRow] = 0.0
	data[data[0::,embarkmentRow]=="Q", embarkmentRow] = 1.0
	data[data[0::,embarkmentRow]=="S", embarkmentRow] = 2.0
	data[data[0::,embarkmentRow]=="", embarkmentRow]=1.0

	#Fill in fare
	if(ifAge == 1):
		fareRow = 6
	else:
		fareRow = 5
	medianFareList = []
	for x in data[0::,fareRow]:
		if(x>0 and x!=''):
			medianFareList.append(float(x))
	data[data[0::,fareRow]=='0.0', fareRow]=float(np.median(medianFareList))
	data[data[0::,fareRow]=='', 6]=float(np.median(medianFareList))

	return data.astype(float)

#Open up training data, csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = csv_file_object.next()
train_age_data=[]   
train_noage_data=[]                       
for row in csv_file_object:      
    if(row[4].isdigit()):
    	train_age_data.append(row)
    else:
    	row.pop(4)
    	train_noage_data.append(row)         
train_age_data = np.array(train_age_data)  
train_noage_data = np.array(train_noage_data)
#Clean and Prepare training data
train_age_data = cleanAndPrep(train_age_data, 1)
train_noage_data = cleanAndPrep(train_noage_data, 0)
#Open up test data, csv file in to a Python object

test_file_object = csv.reader(open('test.csv', 'rU'))
header = test_file_object.next()

test_data=[]                          
for row in test_file_object:      
    row.insert(0,0)
    test_data.append(row)             
test_data = np.array(test_data)  

#Clean and Prepare test data
test_data = cleanAndPrep(test_data, 1)
test_data = test_data[0::, 1:8]

ForestAge = RandomForestClassifier(n_estimators = 100)    
ForestNoAge = RandomForestClassifier(n_estimators = 100)     
ForestAge = ForestAge.fit(train_age_data[0::,1::], train_age_data[0::,0]) 
ForestNoAge = ForestNoAge.fit(train_noage_data[0::,1::], train_noage_data[0::,0]) 
#Output = Forest.predict(test_data)  
train_age_data =train_age_data[0::,1:8]
train_noage_data=train_noage_data[0::,1:7]     
            
OutputAge = ForestAge.predict(train_age_data)
OutputNoAge = ForestNoAge.predict(train_noage_data)
#Prepare output
#test_file_object = csv.reader(open('test.csv', 'rU'))
#header = test_file_object.next()
test_file_object = csv.reader(open('train.csv', 'rU'))
header = test_file_object.next()

#Run RandomForest
header.insert(0, "survived")
open_file_object = csv.writer(open("RandomForest2.csv", "wb"))
open_file_object.writerow(header)
countAge = 0
countNoAge = 0

for row in test_file_object:
	try:
		if(row[4].isdigit()):
			if(OutputAge[countAge]>0):
				row.insert(0, 1)
			else:
				row.insert(0, 0)
			print row
			countAge = countAge+1
		else:
			if(OutputNoAge[countNoAge]>0):
				row.insert(0, 1)
			else:
				row.insert(0, 0)
			print row
			countNoAge = countNoAge+1
		open_file_object.writerow(row)
	except:
		pass


