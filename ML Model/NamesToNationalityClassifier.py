import copy, numpy as np
import random 
from sklearn.utils import shuffle

class NamesToNationalityClassifier:
    
    def __init__(self, examples, labels, possible_labels):
        self.alpha = 0.1
        self.input_dimensions = 27
        self.hidden_dimensions = 500
        self.output_dimensions = len(possible_labels) #124
        self.epsilon_init = 0.12
        self.training_to_validation_ratio = 0.7 # This means 70% of the dataset will be used for training, and 30% is for validation

        self.layer_1_weights = np.random.random((self.hidden_dimensions, self.input_dimensions + 1)) * (2 * self.epsilon_init) - self.epsilon_init
        self.layer_2_weights = np.random.random((self.output_dimensions, self.hidden_dimensions + 1)) * (2 * self.epsilon_init) - self.epsilon_init
        self.hidden_state_weights = np.random.random((self.hidden_dimensions, self.hidden_dimensions)) * (2 * self.epsilon_init) - self.epsilon_init

        self.layer_1_bias = 1
        self.layer_2_bias = 1

        self.num_epoche = 20

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


    '''
        Trains the model based on the training data
    '''
    def train(self):

        for epoche in range(self.num_epoche):
            total_cost = 0

            # Reshuffle the data
            self.serialized_training_examples, self.serialized_training_labels = shuffle(
                self.serialized_training_examples, self.serialized_training_labels)

            for i in range(len(self.serialized_training_examples)):

                # It is a num_char x 27 matrix
                example = self.serialized_training_examples[i]

                # It is a 1D 124 element array
                label = self.serialized_training_labels[i] 

                num_chars = len(example)

                # Stores the hidden state for each letter position.
                letter_pos_to_hidden_state = np.zeros((num_chars + 1, self.hidden_dimensions))

                # Stores the layer 2 values for each letter position
                letter_pos_to_layer_2_values = np.zeros((num_chars, self.hidden_dimensions))

                # Stores the hypothesis for each letter position
                letter_pos_to_hypothesis = np.zeros((num_chars, self.output_dimensions))

                # The hidden state for the first letter position is all 0s.
                letter_pos_to_hidden_state[0] = np.zeros(self.hidden_dimensions)

                letter_pos_to_loss = np.zeros(num_chars)

                example_loss = 0

                # Perform forward propagation
                for j in range(num_chars):

                    # The inputs
                    X = example[j]
                    X_with_bias = np.r_[[self.layer_1_bias], X] # <- We add a bias to the input. It is now a 28 element array
                    hidden_state = letter_pos_to_hidden_state[j - 1]

                    layer_2_values = self.__sigmoid__(np.dot(self.layer_1_weights, X_with_bias) + np.dot(self.hidden_state_weights, hidden_state))

                    # Adding the bias
                    layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values] 

                    hypothesis = self.__sigmoid__(np.dot(self.layer_2_weights, layer_2_values_with_bias))

                    # Update the dictionaries
                    letter_pos_to_layer_2_values[j] = layer_2_values
                    letter_pos_to_hypothesis[j] = hypothesis
                    letter_pos_to_hidden_state[j] = layer_2_values

                    loss = self.__get_cross_entropy__(hypothesis, label)
                    letter_pos_to_loss[j] = loss
                    example_loss += loss

                total_cost += example_loss

                # Perform back propagation through time
                delta_1 = np.zeros((self.hidden_dimensions, self.input_dimensions + 1))
                delta_h = np.zeros((self.hidden_dimensions, self.hidden_dimensions))
                delta_2 = np.zeros((self.output_dimensions, self.hidden_dimensions + 1))

                for j in range(num_chars - 1, -1, -1):
                    X = example[j]
                    hidden_state = letter_pos_to_hidden_state[j]
                    
                    layer_2_values = letter_pos_to_layer_2_values[j]

                    # Adding the bias
                    layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values]
                    
                    hypothesis = letter_pos_to_hypothesis[j]

                    sigma_3 = (hypothesis - label)
                    sigma_2 = np.multiply(np.dot(self.layer_2_weights.T, sigma_3), self.__derivative_sigmoid_given_sigmoid_val__(layer_2_values_with_bias))

                    # We are removing the bias value
                    sigma_2 = sigma_2[1:]

                    # We are not updating the weights of the bias value, so we are setting the changes for the bias weights to 0
                    # We are going to update the weights of the bias value later
                    delta_2 += np.c_[np.zeros(self.output_dimensions), np.dot(np.array([sigma_3]).T, np.array([layer_2_values]))]
                    delta_h += np.dot(sigma_2, np.array([hidden_state]).T)
                    delta_1 += np.c_[np.zeros(self.hidden_dimensions), np.dot(np.array([sigma_2]).T, np.array([X]))]

                self.layer_2_weights -= self.alpha * delta_2
                self.layer_1_weights -= self.alpha * delta_1
                self.hidden_state_weights -= self.alpha * delta_h

                # print('Progress:', (i / len(self.serialized_training_examples)))
            avg_error, accuracy = self.__validate__()
            print('epoche:', epoche, '| avg error:', avg_error, ' | accuracy:', accuracy)

    '''
        It computes how well the model runs based on the validation data
    '''
    def __validate__(self):
        total_cost = 0
        num_correct = 0
        num_examples_ran = 0

        for i in range(len(self.serialized_testing_examples)):

            # It is a num_char x 27 matrix
            example = self.serialized_testing_examples[i]

            # It is a 1D 124 element array
            label = self.serialized_testing_labels[i] 

            num_chars = len(example)

            hypothesis = None
            hidden_state = np.zeros(self.hidden_dimensions)

            total_char_loss = 0

            # Perform forward propagation
            for j in range(num_chars):

                # The inputs
                X = example[j]
                X_with_bias = np.r_[[self.layer_1_bias], X] # <- We add a bias to the input. It is now a 28 element array

                layer_2_values = self.__sigmoid__(np.dot(self.layer_1_weights, X_with_bias) + np.dot(self.hidden_state_weights, hidden_state))

                # Adding the bias
                layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values] 

                hypothesis = self.__sigmoid__(np.dot(self.layer_2_weights, layer_2_values_with_bias))

                hidden_state = layer_2_values

                loss = self.__get_cross_entropy__(hypothesis, label)
                total_char_loss += loss

            # See what was predicted
            if hypothesis is not None:
                label_with_one_index = np.where(label == 1)
                hypothesis_with_max_val_index = np.where(hypothesis == np.amax(hypothesis))

                if label_with_one_index == hypothesis_with_max_val_index:
                    num_correct += 1

                total_cost += total_char_loss

                num_examples_ran += 1

        avg_cost = total_cost / num_examples_ran
        accuracy = num_correct / num_examples_ran

        return avg_cost, accuracy

    def predict(self, name):
        # Serialize the name to a num_char x 27 matrix
        example = self.__serialize__example__(name)

        num_chars = len(example)

        hypothesis = None
        hidden_state = np.zeros(self.hidden_dimensions)

        # Perform forward propagation
        for j in range(num_chars):

            # The inputs
            X = example[j]
            X_with_bias = np.r_[[self.layer_1_bias], X] # <- We add a bias to the input. It is now a 28 element array

            layer_2_values = self.__sigmoid__(np.dot(self.layer_1_weights, X_with_bias) + np.dot(self.hidden_state_weights, hidden_state))

            # Adding the bias
            layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values] 

            hypothesis = self.__sigmoid__(np.dot(self.layer_2_weights, layer_2_values_with_bias))

            hidden_state = layer_2_values

        formatted_hypothesis = []
        for k in range(self.output_dimensions):
            formatted_hypothesis.append((hypothesis[k], self.index_to_label[k]))

        formatted_hypothesis.sort()

        return formatted_hypothesis

    def save_model(self, filename):
        np.savez_compressed(filename, 
            layer_1_weights=self.layer_1_weights, 
            layer_2_weights=self.layer_2_weights, 
            hidden_state_weights=self.hidden_state_weights)

    def load_model_from_file(self, filename):
        data = np.load(filename)
        self.layer_1_weights = data['layer_1_weights']
        self.layer_2_weights = data['layer_2_weights']
        self.hidden_state_weights = data['hidden_state_weights']

    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))

    def __derivative_sigmoid_given_sigmoid_val__(self, sigmoid_value):
	    return sigmoid_value * (1 - sigmoid_value)

    def __softmax__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def __get_cross_entropy__(self, hypothesis, expected_result):
        a = -expected_result
        b = np.log(hypothesis + 1e-15)

        cost = np.multiply(a, b)

        return np.sum(cost)

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
