import random
import itertools
from loaddata import load_dataset
from NNclassifier import NNClassifier
from validator import Validator

def forwardsSelection(datasetpath):
    features, labels = load_dataset(datasetpath)
    NN = NNClassifier()
    callValidator = Validator(NN)

    # randomVal = str(round(random.uniform(50, 100), 1))
    # print("Using no features and \"random\" evaluation, I get an accuracy of " + randomVal + "%\n")
    
    overallMaxAccuracy = str(0)
    maxValue = 0
    maxIndex = 0
    featureLst = []
    resultLst = []
    bestLst = []
    # numOfFeatures = len(features[0])
    
    for i in range(0, len(features[0])):
        featureLst.append(i + 1)
    
    print("Beginning Search.\n")
    resultLst.clear()

    for i in range(0, len(features[0])):
        maxAccuracy = str(0)
        for j in range(0, len(featureLst)):
            tempLst = resultLst.copy()
            tempLst.append(featureLst[j])
            if len(resultLst) == 0:
                tempAccuracy = str(callValidator.evaluate(features, labels, tempLst))
                print("     Using feature(s) " + str(set(tempLst)) + " arruracy is " + tempAccuracy)
            elif len(resultLst) != 0:
                tempAccuracy = str(callValidator.evaluate(features, labels, tempLst))
                print("     Using feature(s) " + str(set(tempLst)) + " arruracy is " + tempAccuracy)

            tempLst.clear()

            if maxAccuracy <= tempAccuracy:
                maxAccuracy = tempAccuracy
                maxValue = featureLst[j]
                maxIndex = j

        resultLst.append(maxValue)
        featureLst.pop(maxIndex)

        if overallMaxAccuracy <= maxAccuracy:
            bestLst = resultLst.copy()
            overallMaxAccuracy = str(maxAccuracy)

        if len(featureLst) != 0:
            print("\nFeature set " + str(set(resultLst)) + " was best, accuracy is " + str(maxAccuracy) + "\n")
        elif len(featureLst) == 0:
            if overallMaxAccuracy <= maxAccuracy:
                print("\nFeature set " + str(set(resultLst)) + " was best, accuracy is " + str(maxAccuracy) + "\n")
            elif overallMaxAccuracy > maxAccuracy:
                print("\nWarning, Accuracy has decreased!)")
                print("Feature set " + str(set(bestLst)) + " was best, accuracy is " + overallMaxAccuracy + "\n")