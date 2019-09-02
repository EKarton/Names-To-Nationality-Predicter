import numpy as np

'''
    This contains helpful functions that are used for ML computations
'''

class ActivationFunctions:

    @staticmethod
    def __sigmoid__(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __derivative_sigmoid_given_sigmoid_val__(sigmoid_value):
	    return sigmoid_value * (1 - sigmoid_value)

    @staticmethod
    def __tanh__(x):
        return np.tanh(x)

    @staticmethod
    def __derivative_tanh_given_tanh_val__(tanh_value):
        return 1.0 - (tanh_value ** 2)

    @staticmethod
    def __softmax__(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    @staticmethod
    def __get_cross_entropy__(hypothesis, expected_result, epsilon=1e-12):
        return -np.sum(np.multiply(expected_result, np.log(hypothesis + epsilon)))