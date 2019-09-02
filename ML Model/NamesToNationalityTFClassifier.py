import copy, numpy as np

class NamesToNationalityTFClassifier:
    def __init__(self, examples, labels, possible_labels):

        self.input_dimensions = 27
        self.hidden_dimensions = 496
        self.output_dimensions = len(possible_labels)
        self.training_to_validation_ratio = 0.7 # This means 70% of the dataset will be used for training, and 30% is for validation

        # We now want to map label to index, and index to label
        self.label_to_index = {}
        self.index_to_label = {}
        
        for i in range(len(possible_labels)):
            label = possible_labels[i]
            self.label_to_index[label] = i
            self.index_to_label[i] = label

        serialized_examples, serialized_labels = self.__serialize_examples_and_labels__(examples, labels)
        num_training_data = int(len(serialized_examples) * self.training_to_validation_ratio)

        self.serialized_training_examples = serialized_examples[:num_training_data]
        self.serialized_training_labels = serialized_labels[:num_training_data]
        self.serialized_testing_examples = serialized_examples[num_training_data:]
        self.serialized_testing_labels = serialized_labels[num_training_data:]

    def train(self):
        pass

    def predict(self, name):
        pass

    def save_model(self, filename):
        pass

    def load_model_from_file(self, filename):
        pass

        '''
        Puts the examples into an array of chars, with each char being a 28 bit array, 
        and labels into a bit array
    '''
    def __serialize_examples_and_labels__(self, examples, labels):
        if len(examples) != len(labels):
            raise Exception('Number of examples does not match number of labels!')

        serialized_examples = []
        serialized_labels = []

        for i in range(len(examples)):
            example = examples[i]
            label = labels[i]
            serialized_example = self.__serialize__example__(example)
            serialized_label = self.__serialize_label__(label)

            if serialized_example is not None and serialized_label is not None:
                serialized_examples.append(serialized_example)
                serialized_labels.append(serialized_label)

        print('serialized', len(serialized_examples), 'examples')
        print('serialized', len(serialized_labels), 'labels')

        return np.array(serialized_examples), np.array(serialized_labels)
                
    '''
        It converts a label into a binary form
        For example, if we have self.label_to_index as:
        {'US': 0, 'Canada': 1, 'Mexico': 2, 'Europe': 3}

        and the label to be 'Mexico', it will return:
        [0, 0, 1, 0].

        Note that the length of the binary array will depend on the number of
        keys in self.label_to_index
    '''
    def __serialize_label__(self, label):
        index = self.label_to_index[label]
        expected_val = np.zeros(self.output_dimensions)
        expected_val[index] = 1
        
        return expected_val

    '''
        Given an example with string 'abc', it will return:
        [
            [1, 0, 0, 0, ..., 0],
            [0, 1, 0, 0, ..., 0],
            [0, 0, 1, 0, ..., 0]
        ]
    '''
    def __serialize__example__(self, example):
        unfiltered_example = example

        # Make letters all lowercase
        # Ex: Mrs. John Smith -> mrs. john smith
        example = example.lower()

        # Remove non-space and non-letter characters
        # Ex: mrs. john smith -> mrs john smith
        filtered_example = ''
        for c in example:
            if 'a' <= c <= 'z' or c == ' ':
                filtered_example += c
        example = filtered_example

        # Remove duplicated spaces
        # Ex: john  smith -> john smith
        example = example.split()
        new_example = ''
        for c in example:
            new_example += c + ' '
        example = new_example[0:-1]

        # Remove names with single letters
        # Ex: john n smith -> john smith
        example = example.split()
        new_example = ''
        for c in example:
            if len(c) > 1:
                new_example += c + ' '
        example = new_example[0:-1]

        # Remove personal titles
        # Ex: mr john smith -> john smith
        personal_titles = set(['dr', 'esq', 'hon', 'jr', 'mr', 'mrs', 'ms', 'messrs', 'mmes', 'msgr', 'prof', 'rev', 'rt', 'sr', 'st'])
        example = example.split()
        new_example = ''
        for c in example:
            if c not in personal_titles:
                new_example += c + ' '
        example = new_example[0:-1]

        # Take only the surname
        # Ex: john smith -> smith
        if len(example) == 0:
            return None

        example = example.split()[-1]

        print('Example:', unfiltered_example, '->', example)

        name_array = []
        for letter in example:
            ascii_code = ord(letter)
            letter_array = np.zeros(self.input_dimensions, )

            if 97 <= ascii_code <= 122:
                letter_array[ascii_code - 97] = 1
            else:
                letter_array[26] = 1

            name_array.append(letter_array)

        return np.array(name_array)