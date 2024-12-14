import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loads the dataset and separates it into features and labels."""
    data = np.loadtxt(file_path)
    labels = data[:, 0] 
    features = data[:, 1:]  
    return features, labels

def plot_scatter(features, labels):
    """Plots a scatter plot with Feature 4 on x-axis and Feature 6 on y-axis."""
    x_values = features[:, 3]  
    y_values = features[:, 5]  
    
    for label, color in zip([1, 2], ['red', 'green']):
        idx = labels == label
        plt.scatter(x_values[idx], y_values[idx], c=color, label=f'Class {label}', alpha=0.7)

    plt.title("Scatter Plot of Feature 4 vs Feature 6")
    plt.xlabel("Feature 4")
    plt.ylabel("Feature 6")
    plt.legend()
    plt.show()

def main():
    file_path = "small-test-dataset.txt"  
    features, labels = load_data(file_path)
    plot_scatter(features, labels)

if __name__ == "__main__":
    main()