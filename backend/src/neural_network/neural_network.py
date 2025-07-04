import random
import numpy as np
from neural_network.layer import Layer

class NeuralNetwork:

    def __init__(self, x_train, y_train, x_test, y_test, epochs, mini_batch_size, learning_rate, filename):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.filename = filename
        self.layers = [Layer(784, 128), Layer(128, 64), Layer(64, 10)]

    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
        
            training_data = list(zip(self.x_train, self.y_train))
            random.shuffle(training_data)
            mini_batches = [training_data[i: i + self.mini_batch_size] for i in range(0, len(training_data), self.mini_batch_size)]
            
            epoch_cost = 0
            batch_count = 0
            
            for mini_batch in mini_batches:
                for layer in self.layers:
                    layer.delta.fill(0)
                    layer.delta_weights.fill(0)
                    layer.delta_biases.fill(0)
                
                batch_cost = 0
                
                for image, target in mini_batch:
                    self.forward_pass(image)
                    actual_output_array = np.zeros(10)
                    actual_output_array[target] = 1
                    batch_cost += self.calculate_cost(self.layers[-1].activations, target)

                    self.backpropagation(image, actual_output_array) 
                                    
                self.update_weights_and_biases()

                epoch_cost += batch_cost
                batch_count += 1
                
                if batch_count % 500 == 0:
                    avg_cost = epoch_cost / (batch_count * self.mini_batch_size)
                    print(f"  Batch {batch_count}, Average cost: {avg_cost:.4f}")
            
            train_acc = self.evaluate_accuracy(self.x_train, self.y_train, 1000)
            test_acc = self.evaluate_accuracy(self.x_test, self.y_test, 1000)
            print(f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        self.save_model()

    def forward_pass(self, input_data):
        current_input = input_data
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].z = np.dot(self.layers[i].weights, current_input) + self.layers[i].biases
            else:
                self.layers[i].z = np.dot(self.layers[i].weights, self.layers[i - 1].activations) + self.layers[i].biases
            self.layers[i].activations = self.sigmoid(self.layers[i].z)
        return np.argmax(self.layers[-1].activations)

    def backpropagation(self, input_image, actual_output_array):
        self.layers[2].delta = (self.layers[2].activations - actual_output_array) * self.d_sigmoid(self.layers[2].z)
        self.layers[1].delta = np.dot(self.layers[2].weights.T, self.layers[2].delta) * self.d_sigmoid(self.layers[1].z)
        self.layers[0].delta = np.dot(self.layers[1].weights.T, self.layers[1].delta) * self.d_sigmoid(self.layers[0].z)
                    
        self.layers[2].delta_weights += np.outer(self.layers[2].delta, self.layers[1].activations)
        self.layers[1].delta_weights += np.outer(self.layers[1].delta, self.layers[0].activations)
        self.layers[0].delta_weights += np.outer(self.layers[0].delta, input_image)
        
        self.layers[2].delta_biases += self.layers[2].delta
        self.layers[1].delta_biases += self.layers[1].delta
        self.layers[0].delta_biases += self.layers[0].delta
    
    def update_weights_and_biases(self):
        for layer in self.layers:
            layer.weights -= (layer.delta_weights / self.mini_batch_size) * self.learning_rate
            layer.biases -= (layer.delta_biases / self.mini_batch_size) * self.learning_rate

    def save_model(self):
        for index, layer in enumerate(self.layers):
            np.save(f"{self.filename}_layer_{index}_weights", layer.weights)
            np.save(f"{self.filename}_layer_{index}_biases", layer.biases)

    def evaluate_accuracy(self, x_data, y_data, max_samples=1000):
        correct = 0
        n_samples = min(len(x_data), max_samples)
        combined = [*zip(x_data, y_data)]
        random.shuffle(combined)
        for i in range(n_samples):
            prediction = self.forward_pass(combined[i][0])
            if prediction == combined[i][1]:
                correct += 1
        return correct / n_samples
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def calculate_cost(self, nn_output, actual_output):
        actual_output_array = np.zeros(10)
        actual_output_array[actual_output] = 1
        return np.sum((nn_output - actual_output_array)**2)



