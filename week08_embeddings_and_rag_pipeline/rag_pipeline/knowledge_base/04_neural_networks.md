# Neural Networks and Deep Learning Basics

## Perceptron and layers
A perceptron is a single artificial neuron: it computes a weighted sum of its inputs plus a bias and applies a step function. A single perceptron can only separate data with a straight line, so it cannot learn the XOR problem. Stacking neurons into layers creates a multi-layer perceptron (MLP) with an input layer, one or more hidden layers, and an output layer. Hidden layers let the network learn non-linear decision boundaries.

## Activation functions
Activation functions add non-linearity. Sigmoid squashes values into 0 to 1 and is used for binary output probabilities, but it suffers from vanishing gradients. Tanh outputs values between -1 and 1 and is zero-centred. ReLU outputs max(0, x); it is fast and the default choice for hidden layers, but neurons can "die" if they always output zero. Leaky ReLU fixes this by allowing a small slope for negative inputs. Softmax turns a vector of scores into probabilities that sum to 1 for multi-class classification.

## Backpropagation
Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule backwards from the output layer to the input layer. The forward pass computes predictions and caches intermediate values; the backward pass uses those cached values to compute gradients efficiently. Gradient checking compares backprop gradients to numerical gradients to verify the implementation.

## Loss functions and optimizers
Mean squared error is the usual loss for regression, and binary cross-entropy is used for binary classification. Categorical cross-entropy is used with softmax for multi-class problems. Optimizers use gradients to update weights. Stochastic gradient descent (SGD) takes steps of size learning_rate times the gradient. Momentum accumulates a velocity to move faster along consistent directions. Adam combines momentum with a per-parameter adaptive learning rate and is a strong default optimizer.
