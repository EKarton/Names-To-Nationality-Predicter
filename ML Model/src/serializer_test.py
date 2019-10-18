import unittest
import math
import numpy as np
from serializer import Serializer

class SerializerTest(unittest.TestCase):
    def test_serialize_example_given_name_should_return_correct_val_1(self):
        name = "Bob Smith"
        serializer = Serializer(['Germany', 'France'])
        serialized_example = serializer.serialize_example(name)

        indexes_with_ones = [2, 15, 2, 0, 19, 13, 9, 20, 8]

        for i in range(len(serialized_example)):
            serialized_char = serialized_example[i]

            self.assertEquals(sum(serialized_char), 1.0)
            self.assertEquals(serialized_char[indexes_with_ones[i]], 1)

    def test_serialize_example_given_name_with_random_spaces_should_return_correct_val_2(self):
        name = "Bob  Smith "
        serializer = Serializer(['Germany', 'France'])
        serialized_example = serializer.serialize_example(name)

        indexes_with_ones = [2, 15, 2, 0, 19, 13, 9, 20, 8]

        for i in range(len(serialized_example)):
            serialized_char = serialized_example[i]

            self.assertEquals(sum(serialized_char), 1.0)
            self.assertEquals(serialized_char[indexes_with_ones[i]], 1)

    def test_serialize_example_given_name_with_unique_chars_should_return_correct_val_3(self):
        name = "Bob  Smáith "
        serializer = Serializer(['Germany', 'France'])
        serialized_example = serializer.serialize_example(name)

        indexes_with_ones = [2, 15, 2, 0, 19, 13, 27, 9, 20, 8]

        for i in range(len(serialized_example)):
            serialized_char = serialized_example[i]

            self.assertEquals(sum(serialized_char), 1.0)
            self.assertEquals(serialized_char[indexes_with_ones[i]], 1)

    def test_serialize_example_given_name_with_random_chars_should_return_correct_val_4(self):
        name = "$$B)ob  Sm#áith "
        serializer = Serializer(['Germany', 'France'])
        serialized_example = serializer.serialize_example(name)

        indexes_with_ones = [2, 15, 2, 0, 19, 13, 27, 9, 20, 8]

        for i in range(len(serialized_example)):
            serialized_char = serialized_example[i]

            self.assertEquals(sum(serialized_char), 1.0)
            self.assertEquals(serialized_char[indexes_with_ones[i]], 1)


if __name__ == '__main__':
    unittest.main()