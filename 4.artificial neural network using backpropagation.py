import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size, hidden_size, output_size = 2, 4, 1
learning_rate, epochs = 0.1, 10000

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_hidden = np.random.uniform(size=(1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_output = np.random.uniform(size=(1, output_size))

for _ in range(epochs):
    hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
    predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)
    
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
    
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output) * learning_rate
    
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer) * learning_rate

output_layer = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)
print("Predicted Output:\n", output_layer)
