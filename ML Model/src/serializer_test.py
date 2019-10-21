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

    def test_serialize_example_given_middle_name_should_return_encoding_without_middle_name(self):
        self.check_serialized_name("Bob Joe Smith", [2, 15, 2, 0, 19, 13, 9, 20, 8])

    def test_serialize_label_given_second_label_should_return_correct_val(self):
        self.check_serialized_label(["Germany", "France"], "France", [0, 1])

    def test_serialize_label_given_nth_label_should_return_correct_val(self):
        self.check_serialized_label(["a", "b", "c", "d"], "c", [0, 0, 1, 0])

    def test_serialize_label_given_unknown_label_should_throw_exception(self):
        with self.assertRaises(Exception):
            self.check_serialized_label(["a", "b", "c", "d"], "e", [0, 0, 0, 0])

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

    def check_serialized_label(self, possible_labels, unserialized_label, expected_serialized_label):
        serializer = Serializer(possible_labels)
        serialized_label = serializer.serialize_label(unserialized_label)

        self.assertEqual(len(serialized_label), len(expected_serialized_label))
        self.assertTrue(np.all(serialized_label == expected_serialized_label))

if __name__ == '__main__':
    unittest.main()