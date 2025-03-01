#-------------------------------------------------------------------------
# AUTHOR: Timothy Tsang
# FILENAME: knn
# SPECIFICATION: Uses k nearest neighbors to predict the class of a sample
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 HOUR
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
error_count = 0
instances = len(db)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for row in db:
        if row != i:
            X.append([float(value) for value in row[:-1]])




    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for row in db:
        if row != i:
            Y.append(float(1 if row[-1] == "ham" else 2))

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(value) for value in i[:-1]]
    real_label = float(1 if i[-1] == "ham" else 2)

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != real_label:
        error_count += 1

#Print the error rate
#--> add your Python code here
error_rate = error_count / instances
print(f"The error rate is {error_rate:.2f}")






