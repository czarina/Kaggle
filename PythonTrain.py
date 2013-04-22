#The first thing to do is to import the relevant packages
# that I will need for my script, 
#these include the Numpy (for maths and arrays)
#and csv for reading and writing csv files
#If i want to use something from this I need to call 
#csv.[function] or np.[function] first

import csv as csv 
import numpy as np

#Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = csv_file_object.next()

data=[]                          #Create a variable called 'data'
for row in csv_file_object:      #Run through each row in the csv file
    data.append(row)             #adding each row to the data variable
data = np.array(data)            #Then convert from a list to an array

women_only_stats = data[0::,3] == "female" #This finds women  

men_only_stats = data[0::,3] != "female"   #This finds men

women_onboard = data[women_only_stats,0].astype(np.float)     
men_onboard = data[men_only_stats,0].astype(np.float)

# Then we find the proportions of them that survived
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

'''
#and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived
'''

#Create passenger bins on fare (set ceiling to 39), gender, class
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

open_file_object = csv.writer(open("genderclasspricebasedmodelpy.csv", "wb"))

# Classify each passenger by gender, class, and fare bins
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


