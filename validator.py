from NNclassifier import NNClassifier
import numpy as np
class Validator:
    def __init__(self,classifier):
        self.classifier = classifier
    #passing in feature data, correct label, and feature subsets
    def evaluate(self, data, labels, featureSubset):
        correctPredictions = 0
        numInstances = len(data)
        currentInstance = 0

        for i in range(numInstances):
            #i gets the index of the instance and feature subset give
            testInstance = []
            for j in featureSubset:
                testInstance.append(data[i][j - 1])
            testLabel = labels[i]
            trainData = []
            trainLabels = []
            for j in range(len(data)):
                if j != i:  
                    trainData.append([data[j][k - 1] for k in featureSubset])
                    trainLabels.append(labels[j])
            trainData = np.array(trainData)
            trainLabels = np.array(trainLabels)
            self.classifier.train(trainData, trainLabels)
            predictedLabel = self.classifier.test(testInstance)

            stringPredictedLabel = str(predictedLabel)
            stringTestLabel = str(testLabel)
            currentInstance += 1
            stringCurrentInstance = str(currentInstance)

            if(predictedLabel == testLabel):
                correctPredictions+=1
            #     print(stringCurrentInstance + " - Predicted Label: " + stringPredictedLabel + " - Test Label: " + stringTestLabel + " - Correct")
            # else:
            #     print(stringCurrentInstance + " - Predicted Label: " + stringPredictedLabel + " - Test Label: " + stringTestLabel + " - Incorrect")
        
        accuracy = (correctPredictions / numInstances) * 100
        return accuracy

