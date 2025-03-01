#-------------------------------------------------------------------------
# AUTHOR: Timothy Tsang
# FILENAME: naive_bayes
# SPECIFICATION: Predicts outcome based on training data
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db = []
with open("weather_training.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for row in db:
    X.append([1 if row[1] == "Sunny" else 2 if row [1] == "Overcast" else 3,
              1 if row[2] == "Hot" else 2 if row[2] == "Mild" else 3,
              1 if row[3] == "High" else 2,
              1 if row[4] == "Strong" else 2 ])


#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in db:
    Y.append(1 if row[5] == "Yes" else 2)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []

with open("weather_test.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTest.append(row)


#Printing the header of the solution
#--> add your Python code here
print("Weather Prediction Results")
print("Outlook| Temperature| Humidty| Wind | PlayTennis Probability")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for row in dbTest:
    test_sample = [
        1 if row[1] == "Sunny" else 2 if row[1] == "Overcast" else 3,
        1 if row[2] == "Hot" else 2 if row[2] == "Mild" else 3,
        1 if row[3] == "High" else 2,
        1 if row[4] == "Strong" else 2,
    ]


    prediction = clf.predict_proba([test_sample])[0]
    predicted_class = "Yes" if prediction[0] > prediction[1] else "No"
    prediction_val = max(prediction)

    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {predicted_class} | {prediction_val:.3f}")




