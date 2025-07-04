import numpy as np

class Layer:
    def __init__(self, in_size: int, out_size: int):
        self.in_size = in_size
        self.out_size = out_size
        self.weights = np.random.randn(out_size, in_size) * np.sqrt(2.0 / in_size)
        self.biases = np.zeros(out_size) 
        self.z = np.zeros(out_size)
        self.delta = np.zeros(out_size)
        self.delta_weights = np.zeros((out_size, in_size))
        self.delta_biases = np.zeros(out_size)
        self.activations = np.zeros(out_size) 