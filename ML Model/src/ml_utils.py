import numpy as np

'''
    This contains useful activation functions
'''
class ActivationFunctions:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative_given_sigmoid_val(sigmoid_value):
	    return sigmoid_value * (1 - sigmoid_value)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative_given_tanh_val(tanh_value):
        return 1.0 - (tanh_value ** 2)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    @staticmethod
    def softmax_derivative(val):
        softmax_val = ActivationFunctions.softmax(val)
        reshaped_softmax_val = softmax_val.reshape(-1,1)
        return np.diagflat(reshaped_softmax_val) - np.dot(reshaped_softmax_val, reshaped_softmax_val.T)

'''
    This contains useful loss functions
'''
class LossFunctions:
    @staticmethod
    def cross_entropy(hypothesis, expected_result, epsilon=1e-12):
        return -np.sum(np.multiply(expected_result, np.log(hypothesis + epsilon)))