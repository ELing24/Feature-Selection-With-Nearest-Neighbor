from loaddata import load_dataset
from NNclassifier import NNClassifier
from validator import Validator

def run(dataset):
    features, labels = load_dataset(dataset)
    return features, labels

small_dataset_path = "small-test-dataset.txt"
large_dataset_path = "large-test-dataset.txt"

features, labels = run(small_dataset_path)
NN = NNClassifier()
callValidator = Validator(NN)
print("Testing Small dataset with features: [3,5,7]")
print("Result should be .89")
result = callValidator.evaluate(features, labels, [3,5,7])
if result == .89:
    print("Test has passed with value: .89" )
else:
    print("Test did not pass with value: " + str(result))
print("\n")
print("Testing Large data with features: [1,15,27]")
print("Result should be .950")
features,labels = run(large_dataset_path)
result = callValidator.evaluate(features, labels, [1,15,27])
if result == .95:
    print("Test has passed with value: .950")
else:
    print("Test did not pass with value " + str(result))

