import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    def __init__(self, x, y, layers):
        self.input = x
        self.y = y
        self.weights = []
        self.l = len(layers)

        for i in range(self.l):
            prev = x.shape[1] if i == 0 else layers[i - 1]
            self.weights.append(np.random.rand(prev, layers[i]))

    def train(self, time, rate):
        for i in range(time):
            self.feedforward()
            self.backprop(rate)

    def feedforward(self):
        self.zs = []
        self.activations = [self.input]
        for i in range(self.l):
            z = np.dot(self.activations[-1], self.weights[i])
            a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)

    def backprop(self, rate):
        error = y - self.activations[-1]
        d_weights2 = np.dot(self.activations[-2].T, error * sigmoid_prime(self.zs[-1]))
        d_weights1 = np.dot(self.activations[-3].T, np.dot(error * sigmoid_prime(self.zs[-1]),
                                                           self.weights[-1].T) * sigmoid_prime(self.zs[-2]))

        self.weights[1] += rate * d_weights2
        self.weights[0] += rate * d_weights1

if __name__ == "__main__":
    X = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    y = np.array([[1],[1],[1],[0],[0],[0]])

    layers = [4, 1]
    time = 100
    learning_rate = 3

    nn = NeuralNetwork(X, y, layers)
    nn.train(time, learning_rate)
    print(nn.activations[-1])