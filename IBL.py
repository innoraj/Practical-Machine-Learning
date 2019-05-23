#k-NN Algorithm Regression
import numpy as np

#Calculate the total sum square
def totalSumSquares(matrix):
    means = np.mean(matrix, axis = 0)
    dist = (means - matrix)**2
    dist = np.sum(dist, axis = 1)
    sumSquares = np.sum(dist, axis = 0)
    return sumSquares

#Calculate the R2 metrics
def rsquareMetric(vector1, vector2, sumSquares):
    residualSum = np.sum((vector1 - vector2)**2)
    rmetric = 1.0 - (residualSum/sumSquares) # rsquare metric used to calculate the accuracy of the IBL algorithm
    return rmetric

#Function to calculate the Rmax
def findRmax(vector, matrix, sumSquares):
    rows, columns = matrix.shape
    rmax, rindex = 0, 0
    for i in range(rows):
        rm = rsquareMetric(vector, matrix[i], sumSquares)
        if rm > rmax:
            rmax, rindex = rm, i
    return rindex

#Classify regression dataset
def classify(testData, trainData, sumSquares):
    output = []
    rows, columns = testData.shape
    for i in range(rows):
        rindex = findRmax(testData[i], trainData, sumSquares)
        output.append(trainData[rindex,-1])
    return output

#Compute the accuracy of the model developed for regression data
def IBLaccuracy(testData, trainData):
    sumSquares = totalSumSquares(trainData[:, 0:-1])
    output = classify(testData[:, 0:-1], trainData[:, 0:-1], sumSquares)
    totalMatching = 0
    for i, x in enumerate(output):
        if x == testData[i,-1]:
            totalMatching += 1
    acc = (float(totalMatching)/float(len(output)))*100.0
    return acc

testData = np.genfromtxt('testData.csv', delimiter = ',')
trainData = np.genfromtxt('trainingData.csv', delimiter = ',')
accuracy = IBLaccuracy(testData, trainData)
print(accuracy)