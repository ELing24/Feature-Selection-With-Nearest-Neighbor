import random
from forwardSelection import forwardsSelection
from backwardselection import backwardSelection

# Group: Akhilesh Genneri - agenn001
# DatasetID: 211
# Small Dataset Results:
#  - Forward: Feature Subset - {3, 5}, Acc: 0.92
#  - Backward: Feature Subset - {2, 4, 5, 7, 10} Acc: 0.82
# Large Dataset Results:
#  - Forward: Feature Subset - {1, 27} Acc: .95
# Titanic Dataset Results:
#  - Forward: Feature Subset - {2} Acc: 0.78
#  - Backward: Feature Subset - {2} Acc: 0.78

print("Welcome to Akhil, Sandeep, and Ethan's Feature Selection Algorithm.\n")
datasetpath = str(input("Type in the name of the file to test : "))
print("\n")

print("Type the number of the algorithm you want to run.\n")
print("1. Forward Selection\n2. Backward Elimination\n3. Bertie's Special Algorithm\n")
algorithmChoice = int(input())
print("\n")

if algorithmChoice == 1:
    forwardsSelection(datasetpath)
elif algorithmChoice == 2:
    backwardSelection(datasetpath)
elif algorithmChoice == 3:
    print()