import unittest
import math
import numpy as np
from names_to_nationality_classifier import NamesToNationalityClassifier

class NamesToNationalityClassifierTest(unittest.TestCase):

    def test_is_hypothesis_correct_should_return_true_case_1(self):
        classifier = NamesToNationalityClassifier([])
        hypothesis = np.array([0.5, 0.4, 0.1])
        label = np.array([1, 0, 0])

        self.assertTrue(classifier.__is_hypothesis_correct__(hypothesis, label))

    def test_is_hypothesis_correct_should_return_true_case_2(self):
        classifier = NamesToNationalityClassifier([])
        hypothesis = np.array([0.4, 0.4, 0.2])
        label_1 = np.array([1, 0, 0])
        label_2 = np.array([1, 0, 0])

        self.assertTrue(classifier.__is_hypothesis_correct__(hypothesis, label_1))
        self.assertTrue(classifier.__is_hypothesis_correct__(hypothesis, label_2))

    def test_is_hypothesis_correct_should_return_false_case_3(self):
        classifier = NamesToNationalityClassifier([])
        hypothesis = np.array([0.5, 0.4, 0.1])
        label = np.array([0, 1, 0])

        self.assertFalse(classifier.__is_hypothesis_correct__(hypothesis, label))

    def test_is_hypothesis_correct_should_return_false_case_4(self):
        classifier = NamesToNationalityClassifier([])
        hypothesis = np.array([0.4, 0.4, 0.2])
        label = np.array([0, 0, 1])

        self.assertFalse(classifier.__is_hypothesis_correct__(hypothesis, label))

if __name__ == '__main__':
    unittest.main()