import unittest
import math
import numpy as np
from ml_utils import ActivationFunctions, LossFunctions

class MLUtilsTest(unittest.TestCase):
    def test_tanh_should_return_correct_value_when_given_single_negative_number(self):
        expected_value = math.tanh(-9)
        actual_value = ActivationFunctions.__tanh__(-9)

        self.assertTrue(abs(actual_value - expected_value) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_negative_numbers_in_array(self):
        expected_values = [math.tanh(i) for i in range(-100, -1)]
        actual_values = ActivationFunctions.__tanh__(np.array([i for i in range(-100, -1)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, len(expected_values)):
            self.assertTrue(abs(actual_values[i] - expected_values[i]) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_negative_numbers_in_2D_array(self):
        expected_values = [ [math.tanh(i) for i in range(-100, -1)] for j in range(0, 10) ]
        actual_values = ActivationFunctions.__tanh__(np.array([ [i for i in range(-100, -1)] for j in range(0, 10)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, 10):
            self.assertEquals(len(expected_values[i]), len(actual_values[i]))

            for j in range(0, len(expected_values[i])):
                self.assertTrue(abs(actual_values[i][j] - expected_values[i][j]) < 0.0000000001)

    def test_tanh_should_return_correct_value_when_given_zero(self):
        expected_value = math.tanh(0)
        actual_value = ActivationFunctions.__tanh__(0)

        self.assertTrue(abs(actual_value - expected_value) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_zeros_in_array(self):
        expected_values = [math.tanh(0) for i in range(-100, -1)]
        actual_values = ActivationFunctions.__tanh__(np.array([0 for i in range(-100, -1)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, len(expected_values)):
            self.assertTrue(abs(actual_values[i] - expected_values[i]) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_zeros_in_2D_array(self):
        expected_values = [ [math.tanh(0) for i in range(-100, -1)] for j in range(0, 10) ]
        actual_values = ActivationFunctions.__tanh__(np.array([ [0 for i in range(-100, -1)] for j in range(0, 10)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, 10):
            self.assertEquals(len(expected_values[i]), len(actual_values[i]))

            for j in range(0, len(expected_values[i])):
                self.assertTrue(abs(actual_values[i][j] - expected_values[i][j]) < 0.0000000001)

    def test_tanh_should_return_correct_value_when_given_single_positive_number(self):
        expected_value = math.tanh(9)
        actual_value = ActivationFunctions.__tanh__(9)

        self.assertTrue(abs(actual_value - expected_value) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_positive_numbers_in_array(self):
        expected_values = [math.tanh(i) for i in range(1, 100)]
        actual_values = ActivationFunctions.__tanh__(np.array([i for i in range(1, 100)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, len(expected_values)):
            self.assertTrue(abs(actual_values[i] - expected_values[i]) < 0.0000000001)

    def test_tanh_should_return_correct_values_when_given_positive_numbers_in_2D_array(self):
        expected_values = [ [math.tanh(i) for i in range(1, 200)] for j in range(0, 10) ]
        actual_values = ActivationFunctions.__tanh__(np.array([ [i for i in range(1, 200)] for j in range(0, 10)]))
        
        self.assertEquals(len(expected_values), len(actual_values))

        for i in range(0, 10):
            self.assertEquals(len(expected_values[i]), len(actual_values[i]))

            for j in range(0, len(expected_values[i])):
                self.assertTrue(abs(actual_values[i][j] - expected_values[i][j]) < 0.0000000001)

    def test_softmax(self):
        input_values = [2.0, 1.0, 0.1]
        expected_values = np.array([0.7, 0.2, 0.1])
        actual_values = ActivationFunctions.__softmax__(input_values)

        for i in range(3):
            self.assertTrue(abs(actual_values[i] - expected_values[i]) < 0.1)

if __name__ == '__main__':
    unittest.main()