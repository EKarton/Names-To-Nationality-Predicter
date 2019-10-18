import unittest
import math
import numpy as np
from serializer import Serializer

class SerializerTest(unittest.TestCase):
    def test_filter_chars_should_return_correct_val_1(self):
        name = "Bob Smith"
        serializer = Serializer(['Germany', 'France'])
        serialized_example, serialized_label = serializer.serialize_examples_and_labels([name], ['Germany'])

        print(serialized_example)
        print(serialized_label)

if __name__ == '__main__':
    unittest.main()