import numpy as np

class Serializer:
    def __init__(self, possible_labels):
        self.allowed_chars_in_filtered_name = [
            ' ', 'a', 'b', 'c', 
            'd', 'e', 'f', 'g', 
            'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 
            'p', 'q', 'r', 's', 
            't', 'u', 'v', 'w', 
            'x', 'y', 'z', 'á', 
            'ã', 'ä', 'ç', 'è', 
            'é', 'ë', 'ï', 'ô', 
            'ö', 'ü', '$', '+',
            '-'
        ]

        self.allowed_chars_in_name = set([
            ' ', 'a', 'b', 'c', 
            'd', 'e', 'f', 'g', 
            'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 
            'p', 'q', 'r', 's', 
            't', 'u', 'v', 'w', 
            'x', 'y', 'z', 'á', 
            'ã', 'ä', 'ç', 'è', 
            'é', 'ë', 'ï', 'ô', 
            'ö', 'ü', '-'
        ])

        self.personal_titles = set([
            'dr', 'esq', 'hon', 'jr', 
            'mr', 'mrs', 'ms', 'messrs', 
            'mmes', 'msgr', 'prof', 'rev', 
            'rt', 'sr', 'st'
        ])

        # Map allowed chars to the index above
        self.allowed_chars_in_filtered_names_to_index = {}
        for i in range(len(self.allowed_chars_in_filtered_name)):
            self.allowed_chars_in_filtered_names_to_index[self.allowed_chars_in_filtered_name[i]] = i

        # We now want to map label to index, and index to label
        self.label_to_index = {}
        self.index_to_label = {}
        
        for i in range(len(possible_labels)):
            label = possible_labels[i]
            self.label_to_index[label] = i
            self.index_to_label[i] = label

        self.input_dimensions = len(self.allowed_chars_in_filtered_name)
        self.target_dimensions = len(possible_labels)

    '''
        Puts the examples into an array of chars, with each char being a 28 bit array, 
        and labels into a bit array
    '''
    def serialize_examples_and_labels(self, examples, labels):
        if len(examples) != len(labels):
            raise Exception('Number of examples does not match number of labels!')

        serialized_examples = []
        serialized_labels = []

        for i in range(len(examples)):
            example = examples[i]
            label = labels[i]
            print(example + " -> " + label)
            serialized_example = self.serialize_example(example)
            serialized_label = self.serialize_label(label)

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
    def serialize_label(self, label):
        index = self.label_to_index[label]
        expected_val = np.zeros(self.target_dimensions)
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
    def serialize_example(self, example):
        filtered_char = self._filter_chars_(example)
        if filtered_char is None:
            return None

        name_array = []
        for letter in filtered_char:
            ascii_code = ord(letter)
            letter_array = np.zeros(self.input_dimensions, )

            if letter in self.allowed_chars_in_filtered_names_to_index:
                letter_array[self.allowed_chars_in_filtered_names_to_index[letter]] = 1
            else:
                raise Exception("Illegal character in name:", letter)

            name_array.append(letter_array)

        return np.array(name_array)

    def _filter_chars_(self, example):
        unfiltered_example = example

        # Make letters all lowercase
        # Ex: Mrs. John Smith -> mrs. john smith
        example = example.lower()

        # Remove non-space and non-letter characters
        # Ex: mrs. john smith -> mrs john smith
        filtered_example = ''
        for c in example:
            if c in self.allowed_chars_in_name:
                filtered_example += c
        example = filtered_example

        # Remove duplicated spaces
        # Ex: john  smith -> john smith
        example = example.split()
        new_example = ''
        for c in example:
            new_example += c + ' '
        example = new_example[0:-1]

        # Remove personal titles
        # Ex: mr john smith -> john smith
        example = example.split()
        new_example = ''
        for c in example:
            if c not in self.personal_titles:
                new_example += c + ' '
        example = new_example[0:-1]

        # Reject those with no characters
        if len(example) == 0 or len(example.split()) == 0:
            return None

        # Reject those whose first or last name is only one letter
        tokenized_example = example.split()
        if len(tokenized_example) == 0 or len(tokenized_example[0]) <= 1 or len(tokenized_example[-1]) <= 1:
            return None

        # Remove names with single letters
        # Ex: john n smith -> john smith
        example = example.split()
        new_example = ''
        for c in example:
            if len(c) > 1:
                new_example += c + ' '
        example = new_example[0:-1]

        tokenized_example = example.split()

        # Needs to contain only first and last name
        # if len(tokenized_example) != 2:
        #     return None


        # Obtain the last name
        # example = tokenized_example[-1]
        # if len(tokenized_example) <= 1:
        #     return None

        # Needs to contain at least the first and last name
        if len(tokenized_example) < 2:
            return None

        # print('OK')

        final_example = ''
        for i in range(len(tokenized_example) - 1):
            final_example += '$' + tokenized_example[i] + '$ '
        final_example += '+' + tokenized_example[-1] + '+'

        # print('Example:', unfiltered_example, '->', final_example, len(final_example))


        return final_example
