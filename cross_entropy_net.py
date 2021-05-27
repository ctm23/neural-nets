class CrossEntNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.l = len(layers)
        self.layers = layers

        for i in range(1, self.l):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
            self.biases.append(np.random.randn(layers[i]))

    def train(self, training_data, time, learning_rate, batch_size, test_data=None):
        training_size = len(training_data)
        self.rate = learning_rate
        for i in range(time):
            np.random.shuffle(training_data)
            mini_x = [training_data[i:i + batch_size, :self.layers[0]] for i in range(0, training_size, batch_size)]
            mini_y = [training_data[i:i + batch_size, self.layers[0]:] for i in range(0, training_size, batch_size)]
            for x, y in zip(mini_x, mini_y):
                self.feedforward(x)
                self.backprop(y)
            if test_data is not None:
                print(f'epoch {i + 1}: {round(self.evaluate(test_data) * 100, 2)}% accurate')

    def feedforward(self, x):
        self.zs = []
        self.activations = [x]
        for i in range(self.l - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)

            a = sigmoid(z)
            self.activations.append(a)

    def backprop(self, y):
        rolling_delta = y - self.activations[-1]
        for i in range(-1, -(self.l), -1):
            derivative = sigmoid_prime(self.zs[i])

            delta_b = rolling_delta * derivative
            delta_w = np.dot(self.activations[i - 1].T, rolling_delta * derivative)
            self.biases[i] += self.rate * np.sum(delta_b, axis=0)
            self.weights[i] += self.rate * delta_w

            rolling_delta = np.dot(rolling_delta * derivative, self.weights[i].T)

    def evaluate(self, test_data):
        self.feedforward(test_data[:, :self.layers[0]])
        digit_yhat = np.argmax(self.activations[-1], axis=1)
        digit_y = np.argmax(test_data[:, self.layers[0]:], axis=1)
        return sum([yhat == y for (yhat, y) in zip(digit_yhat, digit_y)]) / len(test_data)