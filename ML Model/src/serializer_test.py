import unittest
import math
import numpy as np
from serializer import Serializer

class SerializerTest(unittest.TestCase):
    def test_serialize_example_given_name_should_return_correct_val_1(self):
        self.check_serialized_name("Bob Smith", [2, 15, 2, 0, 19, 13, 9, 20, 8])

    def test_serialize_example_given_name_with_random_spaces_should_return_correct_val_2(self):
        self.check_serialized_name("Bob  Smith ", [2, 15, 2, 0, 19, 13, 9, 20, 8])

    def test_serialize_example_given_name_with_unique_chars_should_return_correct_val_3(self):
        self.check_serialized_name("Bob  Smáith ", [2, 15, 2, 0, 19, 13, 27, 9, 20, 8])

    def test_serialize_example_given_name_with_random_chars_should_return_correct_val_4(self):
        self.check_serialized_name("$$B)ob  Sm#áith *", [2, 15, 2, 0, 19, 13, 27, 9, 20, 8])

    def test_serialize_example_given_name_with_pronoun_should_return_correct_val_4(self):
        self.check_serialized_name("Dr. Bob Smith", [2, 15, 2, 0, 19, 13, 9, 20, 8])

    def test_serialize_example_given_single_letter_should_return_correct_val_4(self):
        self.check_serialized_name("Bob C Smith", [2, 15, 2, 0, 19, 13, 9, 20, 8])

    def test_serialize_example_given_single_letter_as_first_name_should_return_none(self):
        self.check_serialized_name("C Bob Smith", None)

    def test_serialize_example_given_single_letter_as_first_name_with_pronoun_should_return_none(self):
        self.check_serialized_name("Mr. C Bob Smith", None)
        
    def test_serialize_example_given_single_letter_as_last_name_should_return_none(self):
        self.check_serialized_name("Bob Joe S", None)

    def check_serialized_name(self, name, expected_indexes_with_ones):
        serializer = Serializer(['Germany', 'France'])
        serialized_example = serializer.serialize_example(name)

        if expected_indexes_with_ones is None:
            self.assertIsNone(serialized_example)
        else:
            self.assertEqual(len(expected_indexes_with_ones), len(serialized_example))

            for i in range(len(expected_indexes_with_ones)):
                serialized_char = serialized_example[i]

                self.assertEqual(sum(serialized_char), 1.0)
                self.assertEqual(serialized_char[expected_indexes_with_ones[i]], 1)

if __name__ == '__main__':
    unittest.main()