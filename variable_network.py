import numpy as np

class leaky_relu:
    def leaky_relu(z):
        return z if z > 0 else 0.1 * z

    def leaky_relu_prime(z):
        return 1.0 if z > 0 else 0.1

    activate = np.vectorize(leaky_relu)
    derivate = np.vectorize(leaky_relu_prime)

class relu:
    def relu(z):
        return max(z, 0.0)

    def relu_prime(z):
        return 1.0 if z > 0 else 0.0

    activate = np.vectorize(relu)
    derivate = np.vectorize(relu_prime)

class sigmoid:
    def activate(z):
        return 1 / (1 + np.exp(-z))

    def derivate(z):
        return sigmoid.activate(z) * (1 - sigmoid.activate(z))


class NeuralNetwork:
    def __init__(self, x, y, layers, activation_function):
        self.input = x
        self.y = y
        self.weights = []
        self.biases = []
        self.l = len(layers)
        self.function = activation_function

        for i in range(self.l):
            prior_nodes = x.shape[1] if i == 0 else layers[i - 1]
            self.weights.append(np.random.randn(prior_nodes, layers[i]))
            self.biases.append(np.random.randn(layers[i]))

    def train(self, time, rate):
        for i in range(time):
            self.feedforward(self.input)
            self.backprop(rate)

    def feedforward(self, x):
        self.zs = []
        self.activations = [x]
        for i in range(self.l):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)

            a = sigmoid.activate(z) if i == self.l - 1 else self.function.activate(z)
            self.activations.append(a)

    def backprop(self, rate):
        rolling_delta = 2 * (self.y - self.activations[-1])
        for i in range(-1, -(self.l + 1), -1):
            derivative = sigmoid.derivate(self.zs[i]) if i == -1 else self.function.derivate(self.zs[i])

            delta_b = rolling_delta * derivative
            delta_w = np.dot(self.activations[i - 1].T, rolling_delta * derivative)
            self.biases[i] += rate * np.sum(delta_b, axis=0)
            self.weights[i] += rate * delta_w

            rolling_delta = np.dot(rolling_delta * derivative, self.weights[i].T)

    def predict(self, x):
        self.feedforward(x)
        return np.around(self.activations[-1],2)


if __name__ == "__main__":
    X = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    y = np.array([[1],[1],[1],[0],[0],[0]])

    layers = [4, 1]
    time = 1000
    learning_rate = .1

    nn = NeuralNetwork(X, y, layers, activation_function=relu)
    nn.train(time, learning_rate)

    print(np.around(nn.activations[-1], 2))
    print('predictions:\n', nn.predict([[1,0,1], [0,0,0]]))

