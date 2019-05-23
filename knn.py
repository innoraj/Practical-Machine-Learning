# k-NN Algorithm

import numpy as np

# Function to calculate Euclidean Distance between training dataset and query dataset
def calculateDistances(a, b):
    dist = (b - a)**2           # Implementing a part of Euclidean distance formula
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    indices = np.argsort(dist) #argsort performs sorting along the given axis
    return dist,indices

# Function to calculate the nearest neighbours with value k
def getNeighbours(testData, trainData, k):
    rows,columns = testData.shape #shape of the array after manipulation
    neighbours = []
    for r in range(rows):
        dist,ind = calculateDistances(testData[r], trainData)
        neighbours.append(ind[:k])
    return neighbours

# Function to classifying whether the dataset belongs to which class(i.e.) benign or malignant
def classify(testData, trainData, k):
    #Sixth column of your data is the classifier. So I am taking the first five columns to compute the neighbours.
    # For a given data point, using the k nearest neighbours and their classifications and
    # using majority voting to classify the data point.
    neighbours = getNeighbours(testData[:, 0:-1], trainData[:, 0:-1], k)
    knearestlist = []
    for i, n in enumerate(neighbours): # enumerate builtin function to use iterator for looping
        x = 0
        for j in n:
            x += trainData[j, -1]
        value = float(x)/float(k)    #k=3 if two neighbours are 1 and the third neighbour is 0, then if you sum them up and divide by k,
        if value >= 0.5:              #the result will be greater than 0.5. In that case the datapoint is classified as 1.
            knearestlist.append(1)
        else:
            knearestlist.append(0)
    return knearestlist

# Compute the accuracy of the kNN algorithm
def computeAccuracy(testData, trainData, k):
    classVector = classify(testData, trainData, k)
    Match_count = 0
    for i, v in enumerate(classVector):
        if v == int(testData[i, -1]):
            Match_count += 1
    accuracy = float(Match_count)/float(len(classVector))
    return accuracy

# Loading training and test dataset
trainData = np.genfromtxt('trainingData2.csv',delimiter=',')
testData = np.genfromtxt('testData2.csv',delimiter=',')
k = 11
Accuracy = (computeAccuracy(testData, trainData, k) * 100)
print('Value of k = ', k, 'Accuracy of the Algorithm  = ', Accuracy)