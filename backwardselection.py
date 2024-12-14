import random
import itertools
from loaddata import load_dataset
from NNclassifier import NNClassifier
from validator import Validator
def backwardSelection(datasetpath):
    features, labels = load_dataset(datasetpath)
    NN = NNClassifier()
    callValidator = Validator(NN)
    randomVal = str(round(random.uniform(50, 100), 1))
    print("Using no features and \"random\" evaluation, I get an accuracy of " + randomVal + "\n")
    
    res = []
    # numOfFeatures = int(numOfFeatures)
    for i in range(0, len(features[0])):
        res.append(i + 1)
    maxValueAndAccuracy = [res, callValidator.evaluate(features,labels, res)]

    #printing for list with all values present
    print("Beginning search.")
    print("\n")
    print("     Using feature(s) " + str(set(res)) + " accuracy is " + str(maxValueAndAccuracy[1]) + "\n")
    print("Feature set " + str(set(res)) + " was best, accuracy is " + str(maxValueAndAccuracy[1]) + "\n")

    #printing all possible combinations
    for i in range(len(features[0])-1,0, -1):
        possibleCombinationsEachLevel = list(itertools.combinations(maxValueAndAccuracy[0], i))
        tmpMaxForLevel = -1
        tmpMaxList = []
        for combination in possibleCombinationsEachLevel:
            # tmpAccuracy = round(random.uniform(50,100),1)
            tmpAccuracy = callValidator.evaluate(features, labels, combination)
            print("     Using feature(s) " + str(set(combination)) + " accuracy is " + str(tmpAccuracy))
            if tmpAccuracy > maxValueAndAccuracy[1]:
                maxValueAndAccuracy = [combination, tmpAccuracy]
            if tmpAccuracy > tmpMaxForLevel:
                tmpMaxForLevel = tmpAccuracy
                tmpMaxList = combination

        print("\nFeature set " + str(set(tmpMaxList)) + " was best, accuracy is " + str(tmpMaxForLevel) + "\n")
        if(tmpMaxForLevel < maxValueAndAccuracy[1]):
            print("(Warning, Accuracy has decreased!)\n")
        
    print("Finished search!! The best feature subset is " + str(set(maxValueAndAccuracy[0])) + ", which has an accuracy of " + str(maxValueAndAccuracy[1]))

