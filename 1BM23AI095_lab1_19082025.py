import math
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, input_size, activation='linear', output_size=1):
        self.input_size = input_size
        self.output_size = output_size
        
        if output_size == 1:
            self.weights = [0.0] * input_size  
            self.bias = 0.0
        else:
            self.weights = np.zeros((output_size, input_size))  
            self.bias = np.zeros(output_size)
        
        # Select activation function
        if activation == 'linear':
            self.activation = self.linear
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'relu':
            self.activation = self.relu
        elif activation == 'softmax':
            if output_size == 1:
                raise ValueError("Softmax requires output_size > 1")
            self.activation = self.softmax
        else:
            raise ValueError("Unsupported activation function")
        self.activation_name = activation
    
    def linear(self, x):
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def tanh(self, x):
        return math.tanh(x)
    
    def relu(self, x):
        return max(0, x)
    
    def softmax(self, x_vec):
        e_x = np.exp(x_vec - np.max(x_vec)) 
        return e_x / e_x.sum()
    
    def forward(self, inputs):
        if self.output_size == 1:
            weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
            return self.activation(weighted_sum)
        else:
            weighted_sum = np.dot(self.weights, inputs) + self.bias
            return self.activation(weighted_sum)


def plot_perceptron_outputs():
    input2 = 1.0
    input1_vals = np.linspace(-10, 10, 400)
    weights = [0.5, -0.6]
    bias = 0.1

    perceptron_linear = Perceptron(2, activation='linear')
    perceptron_linear.weights = weights
    perceptron_linear.bias = bias

    perceptron_sigmoid = Perceptron(2, activation='sigmoid')
    perceptron_sigmoid.weights = weights
    perceptron_sigmoid.bias = bias

    perceptron_tanh = Perceptron(2, activation='tanh')
    perceptron_tanh.weights = weights
    perceptron_tanh.bias = bias

    perceptron_relu = Perceptron(2, activation='relu')
    perceptron_relu.weights = weights
    perceptron_relu.bias = bias

    outputs_linear = []
    outputs_sigmoid = []
    outputs_tanh = []
    outputs_relu = []

    for x1 in input1_vals:
        inputs = [x1, input2]
        outputs_linear.append(perceptron_linear.forward(inputs))
        outputs_sigmoid.append(perceptron_sigmoid.forward(inputs))
        outputs_tanh.append(perceptron_tanh.forward(inputs))
        outputs_relu.append(perceptron_relu.forward(inputs))

    plt.figure(figsize=(10, 6))
    plt.plot(input1_vals, outputs_linear, label='Linear')
    plt.plot(input1_vals, outputs_sigmoid, label='Sigmoid')
    plt.plot(input1_vals, outputs_tanh, label='Tanh')
    plt.plot(input1_vals, outputs_relu, label='ReLU')

    plt.title('Perceptron Outputs with Different Activations')
    plt.xlabel('Input 1 (Input 2 fixed at 1.0)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.show()

def demo_softmax():
    perceptron_softmax = Perceptron(2, activation='softmax', output_size=2)
    perceptron_softmax.weights = np.array([[0.5, -0.6],
                                           [-0.3, 0.8]])
    perceptron_softmax.bias = np.array([0.1, -0.1])

    input2 = 1.0
    input1_vals = np.linspace(-10, 10, 400)

    outputs_softmax_class1 = []
    outputs_softmax_class2 = []

    for x1 in input1_vals:
        inputs = [x1, input2]
        output = perceptron_softmax.forward(inputs) 
        outputs_softmax_class1.append(output[0])
        outputs_softmax_class2.append(output[1])

    plt.figure(figsize=(10, 6))
    plt.plot(input1_vals, outputs_softmax_class1, label='Softmax Class 1 Probability')
    plt.plot(input1_vals, outputs_softmax_class2, label='Softmax Class 2 Probability')
    plt.title('Softmax Perceptron Outputs (Probabilities)')
    plt.xlabel('Input 1 (Input 2 fixed at 1.0)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_perceptron_outputs()

demo_softmax()






