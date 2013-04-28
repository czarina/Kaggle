#Classify training data into bins based on gender, class, and fare
#bucket. Create a survival table calculating mean survival rate per
#bin and make deterministic. Create predictions by finding correct
#bin for each passenger in test data. 

import csv as csv 
import numpy as np

#Exploratory analysis - survival likelihoods by gender, class, etc. 

#Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = csv_file_object.next()

data=[]                          
for row in csv_file_object:      
    data.append(row)             
data = np.array(data)            

women_only_stats = data[0::,3] == "female" 

men_only_stats = data[0::,3] != "female"   

women_onboard = data[women_only_stats,0].astype(np.float)     
men_onboard = data[men_only_stats,0].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

'''
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived
'''

#Begin creation of bins for survival table. Create passenger bins on fare, gender, class
fare_ceiling = 40
data[data[0::,8].astype(np.float) >= fare_ceiling, 8] = fare_ceiling-1.0
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
number_of_classes = 3 #There were 1st, 2nd and 3rd classes on board 

# Define the survival table
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in xrange(number_of_classes):       #search through each class
    for j in xrange(number_of_price_brackets):  #search through each price
        #find survival data for women in ith class and jth price bracket
        women_only_stats = data[(data[0::,3] == "female") \
                       &(data[0::,1].astype(np.float) == i+1) \
                       &(data[0:,8].astype(np.float) >= j*fare_bracket_size) \
                       &(data[0:,8].astype(np.float) < (j+1)*fare_bracket_size), 0]                                                                                                                     

        #find survival data for men in ith class and jth price bracket
        men_only_stats = data[(data[0::,3] != "female")  \
                       &(data[0::,1].astype(np.float) == i+1) \
                       &(data[0:,8].astype(np.float) >= j*fare_bracket_size) \
                       &(data[0:,8].astype(np.float) < (j+1)*fare_bracket_size), 0] 

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) #Women stats
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float)) #Men stats

survival_table[ survival_table != survival_table ] = 0. #set mean of empty bins to 0

# Make table deterministic
survival_table[ survival_table >= 0.5]=1
survival_table[ survival_table<0.5]= 0

# Create predictions from survival table
test_file_object = csv.reader(open('test.csv', 'rU'))
header = test_file_object.next()
header.insert(0, "Survived")
open_file_object = csv.writer(open("genderclasspricebasedmodelpy.csv", "wb"))
open_file_object.writerow(header)
# Draw predictions for test data using the survival table, and output
for row in test_file_object:
    if row[2] == 'female':
        i1 = 0
    else:
        i1 = 1
   
    i2 = int(row[0])-1
    # If no fare data, assume fare correlated with class, i.e., third class is first bin, etc.
    try:
        i3= min(int(float(row[7])/float(fare_bracket_size)), number_of_price_brackets-1)
    except:
        i3 = 3 - float(row[0])

    # Insert survival prediction for each passenger from the bin value in survival table
    row.insert(0, survival_table[i1, i2, i3])

    open_file_object.writerow(row)


