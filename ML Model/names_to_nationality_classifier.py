import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 
from sklearn.utils import shuffle
from ml_utils import ActivationFunctions, LossFunctions

class NamesToNationalityClassifier:
    
    def __init__(self, examples, labels, possible_labels):
        self.alpha = 0.001
        self.input_dimensions = 27
        self.hidden_dimensions = 496
        self.output_dimensions = len(possible_labels)
        self.epsilon_init = 0.12
        self.training_to_validation_ratio = 0.7 # This means 70% of the dataset will be used for training, and 30% is for validation

        self.layer_1_weights = np.random.random((self.hidden_dimensions, self.input_dimensions + 1)) * (2 * self.epsilon_init) - self.epsilon_init
        self.layer_2_weights = np.random.random((self.output_dimensions, self.hidden_dimensions + 1)) * (2 * self.epsilon_init) - self.epsilon_init
        self.hidden_state_weights = np.random.random((self.hidden_dimensions, self.hidden_dimensions)) * (2 * self.epsilon_init) - self.epsilon_init

        self.layer_1_bias = 1
        self.layer_2_bias = 1

        self.num_epoche = 100

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
        Trains the model based on the training data provided.
        It will output a graph.
    '''
    def train(self):

        for epoche in range(self.num_epoche):
            train_avg_error = 0
            train_accuracy = 0

            # Reshuffle the data
            self.serialized_training_examples, self.serialized_training_labels = shuffle(
                self.serialized_training_examples, self.serialized_training_labels)

            for i in range(len(self.serialized_training_examples)):

                # It is a "num_char" x "self.input_dimensions" matrix
                example = self.serialized_training_examples[i]

                # It is a 1D array with "self.output_dimensions" elements
                label = self.serialized_training_labels[i] 

                # Perform forward propagation
                forward_propagation_results = self.__perform_forward_propagation__(example, label)
                letter_pos_to_hypothesis = forward_propagation_results['letter_pos_to_hypothesis']
                letter_pos_to_loss = forward_propagation_results['letter_pos_to_loss']

                # Calculate the train avg error and the train accuracy
                train_avg_error += np.sum(letter_pos_to_loss)
                train_accuracy += 1 if self.__is_hypothesis_correct__(letter_pos_to_hypothesis[-1], label) else 0

                # Perform back propagation
                self.__perform_back_propagation__(example, label, forward_propagation_results)



                # # The gradients that will be computed from the for loop
                # layer_1_weights_gradient = np.zeros((self.hidden_dimensions, self.input_dimensions + 1))
                # layer_2_weights_gradient = np.zeros((self.output_dimensions, self.hidden_dimensions + 1))
                # hidden_weights_gradient = np.zeros((self.hidden_dimensions, self.hidden_dimensions))

                # num_chars = len(example)

                # for j in range(num_chars - 1, -1, -1):
                #     X = example[j]
                #     X_with_bias = np.r_[[self.layer_1_bias], X]
                    
                #     # This is a 1D array with "self.hidden_dimensions" elements
                #     hidden_state = letter_pos_to_hidden_state[j]                    

                #     # This is a 1D array with "self.hidden_dimensions" elements
                #     layer_2_values = letter_pos_to_layer_2_values[j]

                #     # Adding the bias
                #     # This is a 1D array with "self.hidden_dimensions + 1" elements
                #     layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values]

                #     # This is a 1D array with "self.output_dimensions" elements                    
                #     hypothesis = letter_pos_to_hypothesis[j]

                #     # This is a 1D array with "self.output_dimentions" elements
                #     delta_3 = hypothesis - label

                #     # This is a 1D array with "self.hidden_dimensions + 1" elements
                #     delta_2 = np.multiply(np.dot(self.layer_2_weights.T, delta_3), ActivationFunctions.tanh_derivative_given_tanh_val(layer_2_values_with_bias))

                #     # We are removing the bias value
                #     # So now it is a "self.hidden_dimensions" elements
                #     delta_2 = delta_2[1:]

                #     # We are not updating the weights of the bias value, so we are setting the changes for the bias weights to 0
                #     # We are going to update the weights of the bias value later
                #     layer_2_weights_gradient += np.dot(np.array([delta_3]).T, np.array([layer_2_values_with_bias]))
                #     layer_1_weights_gradient += np.dot(np.array([delta_2]).T, np.array([X_with_bias]))
                #     hidden_weights_gradient += np.dot(np.array([delta_2]).T, np.array([hidden_state]))

                # self.layer_2_weights -= self.alpha * layer_2_weights_gradient
                # self.layer_1_weights -= self.alpha * layer_1_weights_gradient
                # self.hidden_state_weights -= self.alpha * hidden_weights_gradient

            train_avg_error /= len(self.serialized_training_examples)
            train_accuracy /= len(self.serialized_training_examples)
            test_avg_error, test_accuracy = self.__validate__()

            # Plot the test_avg_error vs epoche
            plt.subplot(2, 2, 1)
            plt.scatter(epoche, test_avg_error)
            plt.title('Test Avg. Error vs Epoche')

            # Plot the test_accuracy vs epoche
            plt.subplot(2, 2, 2)
            plt.scatter(epoche, test_accuracy)
            plt.title('Test Accuracy vs Epoche')

            # Plot the train_avg_error vs epoche
            plt.subplot(2, 2, 3)
            plt.scatter(epoche, train_avg_error)
            plt.title('Train Avg. Error vs Epoche')

            # Plot the train_accuracy vs epoche
            plt.subplot(2, 2, 4)
            plt.scatter(epoche, train_accuracy)
            plt.title('Train Accuracy vs Epoche')

            # We need to pause so that it will show the graph in realtime
            plt.pause(0.05)

            print('epoche:', epoche, '| test avg error:', test_avg_error, '| test accuracy:', test_accuracy, '|train avg error:', train_avg_error, '|train accuracy:', train_accuracy)

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

            forward_propagation_results = self.__perform_forward_propagation__(example, label)
            letter_pos_to_loss = forward_propagation_results['letter_pos_to_loss']
            letter_pos_to_hypothesis = forward_propagation_results['letter_pos_to_hypothesis']

            if len(letter_pos_to_hypothesis) > 0:
                final_hypothesis = letter_pos_to_hypothesis[-1]

                # Seeing whether the hypothesis is correct
                if self.__is_hypothesis_correct__(final_hypothesis, label):
                    num_correct += 1

                total_cost += np.sum(letter_pos_to_loss)

                num_examples_ran += 1

        avg_cost = total_cost / num_examples_ran
        accuracy = num_correct / num_examples_ran

        return avg_cost, accuracy

    def __is_hypothesis_correct__(self, hypothesis, label):
        label_with_one_index = np.where(label == 1)[0]
        hypothesis_with_max_val_indexes = np.where(hypothesis == np.amax(hypothesis))[0]

        for index in hypothesis_with_max_val_indexes:
            if index in label_with_one_index:
                return True
        return False

    '''
        This function will perform a forward propagation with the serialized version of the example
        and the serialized version of the label.

        The serialized_example needs to be a 2D matrix with size num_char x self.input_dimensions.
        The serialized_label needs to be a 1D array with size self.output_dimentions.

        So this function will return:
        - the loss at each timestep (called 'letter_pos_to_loss')
        - the hidden states at each timestep (called 'letter_pos_to_hidden_state')
        - the layer 2 values at each timestep (called 'letter_pos_to_layer_2_values')
        - the hypothesis at each timestep (called 'letter_pos_to_hypothesis')
        - 
    '''
    def __perform_forward_propagation__(self, serialized_example, serialized_label):
        num_chars = len(serialized_example)

        # Stores the hidden state for each letter position.
        letter_pos_to_hidden_state = np.zeros((num_chars + 1, self.hidden_dimensions))

        # Stores the layer 2 values for each letter position
        letter_pos_to_layer_2_values = np.zeros((num_chars, self.hidden_dimensions))

        # Stores the hypothesis for each letter position
        letter_pos_to_hypothesis = np.zeros((num_chars, self.output_dimensions))

        # The hidden state for the first letter position is all 0s.
        letter_pos_to_hidden_state[0] = np.zeros(self.hidden_dimensions)

        # The loss for each letter position
        letter_pos_to_loss = np.zeros((num_chars, ))

        for j in range(num_chars):
            # The inputs
            X = serialized_example[j]
            X_with_bias = np.r_[[self.layer_1_bias], X] # <- We add a bias to the input. It is now a 28 element array
            hidden_state = letter_pos_to_hidden_state[j - 1]

            layer_2_values = ActivationFunctions.tanh(np.dot(self.layer_1_weights, X_with_bias) + np.dot(self.hidden_state_weights, hidden_state))

            # Adding the bias
            layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values] 

            hypothesis = ActivationFunctions.softmax(np.dot(self.layer_2_weights, layer_2_values_with_bias))

            # Update the dictionaries
            letter_pos_to_layer_2_values[j] = layer_2_values
            letter_pos_to_hypothesis[j] = hypothesis
            letter_pos_to_hidden_state[j] = layer_2_values

            letter_pos_to_loss[j] = LossFunctions.cross_entropy(hypothesis, serialized_label)
        
        return {
            'letter_pos_to_loss': letter_pos_to_loss,
            'letter_pos_to_hidden_state': letter_pos_to_hidden_state,
            'letter_pos_to_layer_2_values': letter_pos_to_layer_2_values,
            'letter_pos_to_hypothesis': letter_pos_to_hypothesis
        }

    '''
        Performs back propagation.
        Note that it requires the results from self.__perform_forward_propagation__() on the same example
        Note that the example needs to be a serialized example, and the label needs to be a serialized label
    '''
    def __perform_back_propagation__(self, serialized_example, serialized_label, forward_propagation_results):
        letter_pos_to_hidden_state = forward_propagation_results['letter_pos_to_hidden_state']
        letter_pos_to_layer_2_values = forward_propagation_results['letter_pos_to_layer_2_values']
        letter_pos_to_hypothesis = forward_propagation_results['letter_pos_to_hypothesis']
        letter_pos_to_loss = forward_propagation_results['letter_pos_to_loss']

        # The gradients
        layer_1_weights_gradient = np.zeros((self.hidden_dimensions, self.input_dimensions + 1))
        layer_2_weights_gradient = np.zeros((self.output_dimensions, self.hidden_dimensions + 1))
        hidden_weights_gradient = np.zeros((self.hidden_dimensions, self.hidden_dimensions))

        num_chars = len(serialized_example)

        for j in range(num_chars - 1, -1, -1):
            X = serialized_example[j]
            X_with_bias = np.r_[[self.layer_1_bias], X]
            
            # This is a 1D array with "self.hidden_dimensions" elements
            hidden_state = letter_pos_to_hidden_state[j]                    

            # This is a 1D array with "self.hidden_dimensions" elements
            layer_2_values = letter_pos_to_layer_2_values[j]

            # Adding the bias
            # This is a 1D array with "self.hidden_dimensions + 1" elements
            layer_2_values_with_bias = np.r_[[self.layer_2_bias], layer_2_values]

            # This is a 1D array with "self.output_dimensions" elements                    
            hypothesis = letter_pos_to_hypothesis[j]

            # This is a 1D array with "self.output_dimentions" elements
            delta_3 = hypothesis - serialized_label

            # This is a 1D array with "self.hidden_dimensions + 1" elements
            delta_2 = np.multiply(np.dot(self.layer_2_weights.T, delta_3), ActivationFunctions.tanh_derivative_given_tanh_val(layer_2_values_with_bias))

            # We are removing the bias value
            # So now it is a "self.hidden_dimensions" elements
            delta_2 = delta_2[1:]

            # We are not updating the weights of the bias value, so we are setting the changes for the bias weights to 0
            # We are going to update the weights of the bias value later
            layer_2_weights_gradient += np.dot(np.array([delta_3]).T, np.array([layer_2_values_with_bias]))
            layer_1_weights_gradient += np.dot(np.array([delta_2]).T, np.array([X_with_bias]))
            hidden_weights_gradient += np.dot(np.array([delta_2]).T, np.array([hidden_state]))

        self.layer_2_weights -= self.alpha * layer_2_weights_gradient
        self.layer_1_weights -= self.alpha * layer_1_weights_gradient
        self.hidden_state_weights -= self.alpha * hidden_weights_gradient

    def predict(self, name):
        # Serialize the name to a num_char x 27 matrix
        example = self.__serialize__example__(name)
        # num_chars = len(example)
        label = np.zeros((self.output_dimensions, ))

        forward_propagation_results = self.__perform_forward_propagation__(example, label)
        letter_pos_to_hypothesis = forward_propagation_results['letter_pos_to_hypothesis']

        if len(letter_pos_to_hypothesis) > 0:
            hypothesis = letter_pos_to_hypothesis[-1]
            formatted_hypothesis = []
            for k in range(self.output_dimensions):
                formatted_hypothesis.append((hypothesis[k], self.index_to_label[k]))

            formatted_hypothesis.sort()

            return formatted_hypothesis
        else:
            raise Exception('Hypothesis cannot be obtained')

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
