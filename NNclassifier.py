import numpy as np
class NNClassifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
    def train(self, data, labels):
        self.training_data = data
        self.training_labels = labels

    def test(self, instance):
        nearest_index = 0
        min_distance = float('inf')

        for i, train_instance in enumerate(self.training_data):
            distance = self.euclidean(instance, train_instance)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        return self.training_labels[nearest_index]
    def euclidean(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
