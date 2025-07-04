from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from neural_network.layer import Layer

# Once the model has been trained, this class loads the weights and biases and predicts the number on a given image
class Predictor:
    def __init__(self):
        self.layers = [Layer(784, 128), Layer(128, 64), Layer(64, 10)]
        self.filename =  Path(__file__).parent.parent.parent / "trained/mnist"
        self.load_weights_and_biases()

    def load_weights_and_biases(self):
        for i in range(len(self.layers)):
            self.layers[i].weights = np.load(f"{self.filename}_layer_{i}_weights.npy")
            self.layers[i].biases = np.load(f"{self.filename}_layer_{i}_biases.npy")
    
    def predict(self, image: ArrayLike):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].z = np.dot(self.layers[i].weights, image) + self.layers[i].biases
            else:
                self.layers[i].z = np.dot(self.layers[i].weights, self.layers[i - 1].activations) + self.layers[i].biases
            self.layers[i].activations = self.sigmoid(self.layers[i].z)
        return self.layers[-1].activations
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
