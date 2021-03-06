import copy
import numpy as np
import random 
from sklearn.utils import shuffle
from ml_utils import ActivationFunctions, LossFunctions
import time
from serializer import Serializer

class NamesToNationalityClassifier:

    def __init__(self, possible_labels, alpha=0.0001, hidden_dimensions=500, l2_lambda = 0.02, momentum=0.9, num_epoche=30):
        self.serializer = Serializer(possible_labels)

        self.alpha = alpha
        self.input_dimensions = self.serializer.input_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.output_dimensions = self.serializer.target_dimensions
        self.training_to_validation_ratio = 0.7 # This means 70% of the dataset will be used for training, and 30% is for validation

        # Weight Initialization
        # We are using the Xavier initialization
        # Reference: https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
        self.weight_init_type = 'X1'
        self.W0 = np.random.randn(self.hidden_dimensions, self.hidden_dimensions) * np.sqrt(1 / self.hidden_dimensions)
        self.W1 = np.random.randn(self.hidden_dimensions, self.input_dimensions + 1) * np.sqrt(1 / (self.input_dimensions + 1))
        self.W2 = np.random.randn(self.output_dimensions, self.hidden_dimensions + 1) * np.sqrt(1 / (self.hidden_dimensions + 1))

        # Momentum and regularization
        self.l2_lambda = l2_lambda # The lambda for L2 regularization
        self.momentum = momentum
        self.W0_velocity = np.zeros((self.hidden_dimensions, self.hidden_dimensions))
        self.W1_velocity = np.zeros((self.hidden_dimensions, self.input_dimensions + 1))
        self.W2_velocity = np.zeros((self.output_dimensions, self.hidden_dimensions + 1))

        # Bias values
        self.layer_1_bias = 1
        self.layer_2_bias = 1

        # Num epoche
        self.num_epoche = num_epoche

        self.serialized_training_examples = []
        self.serialized_training_labels = []
        self.serialized_testing_examples = []
        self.serialized_testing_labels = []

    
    def add_training_examples(self, examples, labels):
        serialized_examples, serialized_labels = self.serializer.serialize_examples_and_labels(examples, labels) #self.__serialize_examples_and_labels__(examples, labels)
        num_training_data = int(len(serialized_examples) * self.training_to_validation_ratio)

        self.serialized_training_examples = serialized_examples[:num_training_data]
        self.serialized_training_labels = serialized_labels[:num_training_data]
        self.serialized_testing_examples = serialized_examples[num_training_data:]
        self.serialized_testing_labels = serialized_labels[num_training_data:]

    '''
        Trains the model based on the training data provided.
        It will output a dictionary with the following keys:
        {
            'epoche_to_train_avg_error': the train avg error per epoche,
            'epoche_to_test_avg_error': the test avg error per epoche,
            'epoche_to_train_accuracy': the train accuracy per epoche,
            'epoche_to_test_accuracy': the test accuracy per epoche
        }
    '''
    def train(self):
        print("Training...")
        print(self)

        epoche_to_train_avg_error = np.zeros((self.num_epoche, ))
        epoche_to_test_avg_error = np.zeros((self.num_epoche, ))
        epoche_to_train_accuracy = np.zeros((self.num_epoche, ))
        epoche_to_test_accuracy = np.zeros((self.num_epoche, ))

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

            epoche_to_train_avg_error[epoche] = train_avg_error / len(self.serialized_training_examples)
            epoche_to_train_accuracy[epoche] = train_accuracy / len(self.serialized_training_examples)

            test_avg_error, test_accuracy, test_runnable_ratio = self.__validate__()
            epoche_to_test_accuracy[epoche] = test_accuracy
            epoche_to_test_avg_error[epoche] = test_avg_error

            print(epoche, epoche_to_train_avg_error[epoche], epoche_to_test_avg_error[epoche], epoche_to_train_accuracy[epoche], epoche_to_test_accuracy[epoche], test_runnable_ratio, time.time())

        return {
            'epoche_to_train_avg_error': epoche_to_train_avg_error,
            'epoche_to_test_avg_error': epoche_to_test_avg_error,
            'epoche_to_train_accuracy': epoche_to_train_accuracy,
            'epoche_to_test_accuracy': epoche_to_test_accuracy
        }

    '''
        Trains an example with a label.
        The example is a name (like "Bob Smith") and its label is a country name (ex: "Canada")
    '''
    def train_example(self, example, label):
        serialized_example = self.serializer.serialize_example(example)
        serialized_label = self.serializer.serialize_label(label)

        # Perform forward propagation
        forward_propagation_results = self.__perform_forward_propagation__(serialized_example, serialized_label)

        # Perform back propagation
        self.__perform_back_propagation__(serialized_example, serialized_label, forward_propagation_results)

    '''
        It computes how well the model runs based on the validation data
        It returns the avg. error and accuracy rate
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
        runnable_examples_ratio = num_examples_ran / len(self.serialized_testing_examples)

        return avg_cost, accuracy, runnable_examples_ratio

    def __is_hypothesis_correct__(self, hypothesis, label):
        return np.argmax(hypothesis, axis=0) == np.argmax(label, axis=0)

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
        letter_pos_to_h0 = np.zeros((num_chars + 1, self.hidden_dimensions))

        # Stores the layer 2 values for each letter position
        letter_pos_to_h1 = np.zeros((num_chars, self.hidden_dimensions))

        # Stores the hypothesis for each letter position
        letter_pos_to_h2 = np.zeros((num_chars, self.output_dimensions))

        # The hidden state for the first letter position is all 0s.
        letter_pos_to_h0[0] = np.zeros(self.hidden_dimensions)

        # The loss for each letter position
        letter_pos_to_loss = np.zeros((num_chars, ))

        for j in range(num_chars):
            # The inputs
            X = serialized_example[j]
            X_with_bias = np.r_[[self.layer_1_bias], X] # <- We add a bias to the input. It is now a 28 element array
            h0 = letter_pos_to_h0[j]

            y1 = np.dot(self.W1, X_with_bias) + np.dot(self.W0, h0)
            h1 = ActivationFunctions.tanh(y1)

            # Adding the bias
            h1_with_bias = np.r_[[self.layer_2_bias], h1]

            y2 = np.dot(self.W2, h1_with_bias)
            h2 = ActivationFunctions.softmax(y2)

            # Update the dictionaries
            letter_pos_to_h1[j] = h1
            letter_pos_to_h2[j] = h2
            letter_pos_to_h0[j + 1] = h1

            letter_pos_to_loss[j] = LossFunctions.cross_entropy(h2, serialized_label)
        
        return {
            'letter_pos_to_loss': letter_pos_to_loss,
            'letter_pos_to_hidden_state': letter_pos_to_h0,
            'letter_pos_to_layer_2_values': letter_pos_to_h1,
            'letter_pos_to_hypothesis': letter_pos_to_h2
        }

    '''
        Performs back propagation.
        Note that it requires the results from self.__perform_forward_propagation__() on the same example
        Note that the example needs to be a serialized example, and the label needs to be a serialized label
    '''
    def __perform_back_propagation__(self, serialized_example, serialized_label, forward_propagation_results):
        letter_pos_to_h0 = forward_propagation_results['letter_pos_to_hidden_state']
        letter_pos_to_h1 = forward_propagation_results['letter_pos_to_layer_2_values']
        letter_pos_to_h2 = forward_propagation_results['letter_pos_to_hypothesis']
        letter_pos_to_loss = forward_propagation_results['letter_pos_to_loss']

        # The loss gradients w.r.t W0, W1, W2
        dL_dW0 = np.zeros((self.hidden_dimensions, self.hidden_dimensions))
        dL_dW1 = np.zeros((self.hidden_dimensions, self.input_dimensions + 1))
        dL_dW2 = np.zeros((self.output_dimensions, self.hidden_dimensions + 1))

        num_chars = len(serialized_example)

        for j in range(num_chars - 1, -1, -1):
            X = serialized_example[j]
            X_with_bias = np.r_[[self.layer_1_bias], X]
            
            # This is a 1D array with "self.hidden_dimensions" elements
            h0 = letter_pos_to_h0[j]                    

            # This is a 1D array with "self.hidden_dimensions" elements
            h1 = letter_pos_to_h1[j]

            # Adding the bias
            # This is a 1D array with "self.hidden_dimensions + 1" elements
            h1_with_bias = np.r_[[self.layer_2_bias], h1]

            # This is a 1D array with "self.output_dimensions" elements                    
            h2 = letter_pos_to_h2[j]

            # This is a 1D array with "self.output_dimentions" elements
            # This is the derivative of y with respect to the cross entropy score
            dL_dY2 = h2 - serialized_label

            # This is a 1D array with "self.hidden_dimensions + 1" elements
            dL_dH1 = np.dot(dL_dY2.T, self.W2)
            dL_dY1 = np.multiply(dL_dH1, ActivationFunctions.tanh_derivative_given_tanh_val(h1_with_bias))

            # We are removing the bias value
            # So now it is a "self.hidden_dimensions" elements
            dL_dY1 = dL_dY1[1:]

            # We are not updating the weights of the bias value, so we are setting the changes for the bias weights to 0
            # We are going to update the weights of the bias value later
            dL_dW0 += np.dot(np.array([dL_dY1]).T, np.array([h0]))
            dL_dW1 += np.dot(np.array([dL_dY1]).T, np.array([X_with_bias]))
            dL_dW2 += np.dot(np.array([dL_dY2]).T, np.array([h1_with_bias]))

        # Add regularization
        dL_dW0 += self.l2_lambda * self.W0
        dL_dW1 += self.l2_lambda * self.W1
        dL_dW2 += self.l2_lambda * self.W2

        # Add the velocity
        self.W0_velocity = (self.momentum * self.W0_velocity) + (self.alpha * dL_dW0)
        self.W1_velocity = (self.momentum * self.W1_velocity) + (self.alpha * dL_dW1)
        self.W2_velocity = (self.momentum * self.W2_velocity) + (self.alpha * dL_dW2)

        # Update weights
        self.W0 -= self.W0_velocity
        self.W1 -= self.W1_velocity
        self.W2 -= self.W2_velocity

    def predict(self, name):
        # Serialize the name to a num_char x 27 matrix
        example = self.serializer.serialize_example(name)
        # num_chars = len(example)
        label = np.zeros((self.output_dimensions, ))

        forward_propagation_results = self.__perform_forward_propagation__(example, label)
        letter_pos_to_y2 = forward_propagation_results['letter_pos_to_hypothesis']

        if len(letter_pos_to_y2) > 0:
            hypothesis = ActivationFunctions.softmax(letter_pos_to_y2[-1])
            formatted_hypothesis = []
            for k in range(self.output_dimensions):
                formatted_hypothesis.append((hypothesis[k], self.serializer.index_to_label[k]))

            formatted_hypothesis.sort(reverse=True)

            return formatted_hypothesis
        else:
            raise Exception('Hypothesis cannot be obtained')

    def save_model(self, filename):
        np.savez_compressed(filename, 
            layer_1_weights=self.W1, 
            layer_2_weights=self.W2, 
            hidden_state_weights=self.W0)

    def load_model_from_file(self, filename):
        data = np.load(filename)
        self.W1 = data['layer_1_weights']
        self.W2 = data['layer_2_weights']
        self.W0 = data['hidden_state_weights']

    def __str__(self):
        description = "RNN with learning rate: {}, momentum: {}, L2 reg. rate: {}, Weight Init. Type: {}, Num. Epoche: {}" 
        return description.format(self.alpha, 
                                  self.momentum, 
                                  self.l2_lambda, 
                                  self.weight_init_type, 
                                  self.num_epoche)
