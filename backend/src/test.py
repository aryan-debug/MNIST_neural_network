import numpy as np
from neural_network.predictor import Predictor
from pathlib import Path

def read_mnist_dataset():
    import gzip, pickle
    f = gzip.open(Path(__file__).parent / "../mnist.pkl.gz", 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data
    return np.array(x_train).reshape((1, 784))/255, np.array(y_train), np.array(x_test).reshape((-1, 784))/255, np.array(y_test)

def main():
    _, _, x_test, y_test = read_mnist_dataset()

    predictor = Predictor()
    for i in range(10):
        print(predictor.predict(x_test[i]))
        print(y_test[i])

main()