import numpy as np

# Assuming you already have a nearest neighbor classifier and leave-one-out validator implemented
def nearest_neighbor_classifier(training_data, test_instance, selected_features):
    """
    Classifies a test instance based on the nearest neighbor approach.
    """
    min_distance = float('inf')
    predicted_label = None
    for instance in training_data:
        distance = np.linalg.norm(test_instance[selected_features] - instance[selected_features])
        if distance < min_distance:
            min_distance = distance
            predicted_label = instance[-1]  # Assuming the last column is the label
    return predicted_label

def leave_one_out_validator(data, selected_features):
    """
    Leave-One-Out cross-validation to compute the accuracy of a feature subset.
    """
    correct_predictions = 0
    for i in range(len(data)):
        training_data = np.delete(data, i, axis=0)
        test_instance = data[i]
        predicted_label = nearest_neighbor_classifier(training_data, test_instance, selected_features)
        if predicted_label == test_instance[-1]:  # Compare predicted and actual label
            correct_predictions += 1
    return correct_predictions / len(data)

# Feature Search Algorithms
def forward_selection(data):
    """
    Forward Selection Algorithm.
    """
    best_features = []
    best_overall_accuracy = 0
    num_features = data.shape[1] - 1  # Excluding the label column

    for i in range(1, num_features + 1):
        feature_to_add = None
        best_accuracy = 0

        for feature in range(1, num_features + 1):
            if feature not in best_features:
                current_features = best_features + [feature]
                accuracy = leave_one_out_validator(data, current_features)
                print(f"Using feature(s) {current_features}, accuracy is {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_add = feature

        if feature_to_add:
            best_features.append(feature_to_add)
            print(f"Feature set {best_features} was best, accuracy is {best_accuracy:.4f}")
            best_overall_accuracy = best_accuracy

    return best_features, best_overall_accuracy

def backward_elimination(data):
    """
    Backward Elimination Algorithm.
    """
    num_features = data.shape[1] - 1  # Excluding the label column
    best_features = list(range(1, num_features + 1))
    best_overall_accuracy = leave_one_out_validator(data, best_features)

    while len(best_features) > 1:
        feature_to_remove = None
        best_accuracy = 0

        for feature in best_features:
            current_features = [f for f in best_features if f != feature]
            accuracy = leave_one_out_validator(data, current_features)
            print(f"Using feature(s) {current_features}, accuracy is {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                feature_to_remove = feature

        if feature_to_remove:
            best_features.remove(feature_to_remove)
            print(f"Feature set {best_features} was best, accuracy is {best_accuracy:.4f}")
            best_overall_accuracy = best_accuracy

    return best_features, best_overall_accuracy

# Main Function to Run
def main():
    # Load and preprocess the dataset (Titanic or others)
    # Assuming the dataset is in NumPy array format after preprocessing
    # Each row: [Feature1, Feature2, ..., FeatureN, ClassLabel]
    data = np.loadtxt("preprocessed_titanic.txt")  # Replace with actual file path

    print("Running Forward Selection...")
    forward_features, forward_accuracy = forward_selection(data)
    print(f"Forward Selection Results: Features {forward_features}, Accuracy {forward_accuracy:.4f}")

    print("\nRunning Backward Elimination...")
    backward_features, backward_accuracy = backward_elimination(data)
    print(f"Backward Elimination Results: Features {backward_features}, Accuracy {backward_accuracy:.4f}")

if __name__ == "__main__":
    main()
