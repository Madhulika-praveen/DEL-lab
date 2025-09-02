import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

class Neuron:
    def __init__(self, weights, bias, activation="sigmoid"):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'tanh':
            return tanh(z)
        elif self.activation == 'relu':
            return relu(z)
        else:
            raise ValueError("Unknown activation function")

inputs = np.array([0.5, -1.2, 3.0])
weights = [0.4, -0.6, 0.3]
bias = 0.1

x = np.linspace(-5, 5, 400)

for act in ['sigmoid', 'tanh', 'relu']:
    neuron = Neuron(weights, bias, act)
    z_value = np.dot(inputs, weights) + bias 
    output = neuron.forward(inputs)

    if act == 'sigmoid':
        y = sigmoid(x)
    elif act == 'tanh':
        y = tanh(x)
    else:
        y = relu(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'{act} function', color='blue')
    plt.scatter(z_value, output, s=100, color='red', label=f'Output = {output:.4f}')
    plt.title(f"Activation Function: {act}")
    plt.xlabel("Input to activation function (z)")
    plt.ylabel("Activation output")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Activation = {act:7s} --> Output = {output:.4f}')
