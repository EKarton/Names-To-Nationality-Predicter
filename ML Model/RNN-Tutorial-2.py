import copy, numpy as np
import random

##########################################################################
############################ Getting the Data ############################
##########################################################################

def get_countries():
	country_id_to_country_name = {}

	with open('countries.csv') as countries_file_reader:

		line = countries_file_reader.readline()
		while line:
			tokenized_line = line.split(',')
			if len(tokenized_line) == 3:
				country_id = int(tokenized_line[0])
				country_name = tokenized_line[1]
				nationality = tokenized_line[2]

				country_id_to_country_name[country_id] = (country_name, nationality)

			line = countries_file_reader.readline()

	return country_id_to_country_name

def get_records():
	records = []
	with open('records.csv') as reader:

		line = reader.readline()
		while line:
			tokenized_line = line.split(',')

			if len(tokenized_line) == 3:
				name = tokenized_line[1]
				country_of_birth_id = int(tokenized_line[2])
				records.append((name, country_of_birth_id))

			line = reader.readline()

	return records

def get_dataset():
	country_id_to_country_name = get_countries()
	country_id_to_reduced_id = {}
	countries_dataset = []

	i = 0
	for country_id in country_id_to_country_name:
		country_id_to_reduced_id[country_id] = i
		countries_dataset.append(country_id_to_country_name[country_id])
		i += 1

	records = get_records()

	records_dataset = []

	for record in records:
		name = record[0]
		country_of_birth_id = record[1]
		reduced_country_of_birth_id = country_id_to_reduced_id[country_of_birth_id]

		records_dataset.append((name, reduced_country_of_birth_id))

	return (countries_dataset, records_dataset)

##########################################################################
############################ Running the RNN #############################
##########################################################################

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def derivative_sigmoid_given_sigmoid_val(sigmoid_value):
	return sigmoid_value * (1 - sigmoid_value)

def get_binary_cross_entropy(hypothesis, expected_result):
	a = -expected_result
	b = np.log(hypothesis + 1e-15)
	c = (1 - expected_result)
	d = np.log(1 - hypothesis + 1e-15)

	cost = np.multiply(a, b) - np.multiply(c, d)
	# cost = np.xlogy(-expected_result, hypothesis) - np.xlogy(1 - expected_result, 1 - hypothesis)

	return np.sum(cost)

def main():
	raw_countries, raw_records = get_dataset()
	countries = np.asarray(raw_countries)
	records = np.asarray(raw_records)

	'''
	The psuedocode for the RNN:

	rnn = RNN()
	ff = FeedForwardNN()
	hidden_state = [0, 0, 0, 0]

	for word in sentence:
		output, hidden_state = rnn(word, hidden_state)

	prediction = ff(output)
	'''

	'''
		Now, we are going to convert the expected values in each training set into a 1 x 124 matrix,
		We are also going to make each name in lowercase
		We are also going to convert each letter into a 1 x 27 matrix, where [1, 0, ..., 0] represents 'a'
	'''
	new_records = []
	for record in records:
		name = record[0].lower()
		country_index = int(record[1])

		expected_val = np.zeros((124,))
		expected_val[country_index] = 1

		name_array = []

		for letter in name:
			ascii_code = ord(letter)
			letter_array = np.zeros((27,))

			if 97 <= ascii_code <= 122:
				letter_array[ascii_code - 97] = 1
			else:
				letter_array[26] = 1

			name_array.append(letter_array)

		new_records.append((np.array(name_array), expected_val))

	records = np.array(new_records)

	alpha = 0.1
	input_dimensions = 27
	hidden_dimensions = 200
	output_dimensions = 124

	# This is a 27 x 200 matrix
	# Layer 1 represents the input layer
	layer_1_weights = 2 * np.random.random((input_dimensions, hidden_dimensions)) - 1 #np.random.rand(input_dimensions, hidden_dimensions)

	# This is a 200 x 124 matrix
	# Layer 2 represents the hidden layer
	layer_2_weights = 2 * np.random.random((hidden_dimensions, output_dimensions)) - 1 #np.random.rand(hidden_dimensions, output_dimensions)

	# This is a 200 x 200 matrix
	hidden_state_weights = 2 * np.random.random((hidden_dimensions, hidden_dimensions)) - 1 # np.random.rand(hidden_dimensions, hidden_dimensions)

	# This is the bias for the first layer
	bias_1 = 1

	# This is the bias for the second layer
	bias_2 = 1

	for _ in range(100):

		# Train on the dataset
		for i in range(len(records[0:499])):
			print('Training', i, 'vs', len(records))
			record = records[i]
			name = record[0]
			num_chars = len(name)
			y = np.array([record[1]])
			
			# Stores the hidden state for each letter position.
			letter_pos_to_hidden_state = np.zeros((num_chars + 1, 1, 200))

			# Stores the layer 2 values for each letter position
			letter_pos_to_layer_2_values = np.zeros((num_chars, 1, 200))

			# Stores the hypothesis for each letter position
			letter_pos_to_hypothesis = np.zeros((num_chars, 1, 124))

			# The hidden state for the first letter position is all 0s.
			letter_pos_to_hidden_state[0] = np.zeros((1, 200))

			letter_pos_to_loss = np.zeros(num_chars)

			overall_error = 0

			# Perform forward propagation
			for i in range(num_chars):
				
				# The inputs
				letter_array = name[i]
				X = np.array([letter_array]) 
				hidden_state = letter_pos_to_hidden_state[i - 1]

				# Perform forward propagation
				layer_2_values = sigmoid(np.dot(X, layer_1_weights) + np.dot(hidden_state, hidden_state_weights) + bias_1)
				hypothesis = sigmoid(np.dot(layer_2_values, layer_2_weights) + bias_2)

				# Update the dictionaries
				letter_pos_to_layer_2_values[i] = layer_2_values
				letter_pos_to_hypothesis[i] = hypothesis
				np.append(letter_pos_to_hidden_state, layer_2_values)

				loss = get_binary_cross_entropy(hypothesis, y)
				letter_pos_to_loss[i] = loss
				overall_error += loss

			# print('letter_pos_to_hidden_state:', letter_pos_to_hidden_state.shape)
			# print('letter_pos_to_layer_2_values:', letter_pos_to_layer_2_values.shape)
			# print('letter_pos_to_loss:', letter_pos_to_loss)
			# print('letter_pos_to_hypothesis:', letter_pos_to_hypothesis.shape)

			# Perform back propagation through time
			delta_h = np.zeros((hidden_dimensions, hidden_dimensions))
			delta_1 = np.zeros((input_dimensions, hidden_dimensions))
			delta_2 = np.zeros((hidden_dimensions, output_dimensions))

			for i in range(len(name) - 1, -1, -1):
				letter_array = name[i]
				X = np.array([letter_array])
				hidden_state = letter_pos_to_hidden_state[i]
				layer_2_values = letter_pos_to_layer_2_values[i]
				hypothesis = letter_pos_to_hypothesis[i]
				
				sigma_3 = (hypothesis - y).T
				# print('sigma_3.shape', sigma_3.shape)

				# print('np.dot(layer_2_weights, sigma_3).shape:', np.dot(layer_2_weights, sigma_3).shape)
				# print('derivative_sigmoid_given_sigmoid_val(layer_2_values).T.shape:', derivative_sigmoid_given_sigmoid_val(layer_2_values).T.shape)
				sigma_2 = np.multiply(np.dot(layer_2_weights, sigma_3), derivative_sigmoid_given_sigmoid_val(layer_2_values).T)
				# print('sigma_2.shape', sigma_2.shape)

				# print('layer_2_values.shape:', layer_2_values.shape)

				delta_2 += np.dot(sigma_3, layer_2_values).T
				# print('delta_2.shape:', delta_2.shape)

				# print('hidden_state.shape:', hidden_state.shape)

				delta_h += np.dot(sigma_2, hidden_state)
				# print('delta_h.shape:', delta_h.shape)

				# print('X.shape:', X.shape)
				delta_1 += np.dot(sigma_2, X).T
				# print('delta_1.shape:', delta_1.shape)

			layer_1_weights -= alpha * delta_1
			layer_2_weights -= alpha * delta_2
			hidden_state_weights -= alpha * delta_h

		# Perform one hypothesis test on a random dataset
		random_record_index = random.randint(0, 499)
		random_record = records[random_record_index]

		name = random_record[0]
		num_chars = len(name)
		y = np.array([random_record[1]])
		
		# Stores the hidden state for each letter position.
		letter_pos_to_hidden_state = np.zeros((num_chars + 1, 1, 200))

		# Stores the layer 2 values for each letter position
		letter_pos_to_layer_2_values = np.zeros((num_chars, 1, 200))

		# Stores the hypothesis for each letter position
		letter_pos_to_hypothesis = np.zeros((num_chars, 1, 124))

		# The hidden state for the first letter position is all 0s.
		letter_pos_to_hidden_state[0] = np.zeros((1, 200))

		letter_pos_to_loss = np.zeros(num_chars)

		overall_error = 0

		# Perform forward propagation
		for i in range(num_chars):
			
			# The inputs
			letter_array = name[i]
			X = np.array([letter_array]) 
			hidden_state = letter_pos_to_hidden_state[i - 1]

			# Perform forward propagation
			layer_2_values = sigmoid(np.dot(X, layer_1_weights) + np.dot(hidden_state, hidden_state_weights) + bias_1)
			hypothesis = sigmoid(np.dot(layer_2_values, layer_2_weights) + bias_2)

			# Update the dictionaries
			letter_pos_to_layer_2_values[i] = layer_2_values
			letter_pos_to_hypothesis[i] = hypothesis
			np.append(letter_pos_to_hidden_state, layer_2_values)

			loss = get_binary_cross_entropy(hypothesis, y)
			letter_pos_to_loss[i] = loss
			overall_error += loss

		print('hypothesis[-1]:', hypothesis[-1])
		print('y:', y)
		print('overall_error:', overall_error)
		input('Press enter to continue')
main()


