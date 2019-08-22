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

	print(countries_dataset)

	records = get_records()

	records_dataset = []

	for record in records:
		name = record[0]
		country_of_birth_id = record[1]
		reduced_country_of_birth_id = country_id_to_reduced_id[country_of_birth_id]

		records_dataset.append((name, reduced_country_of_birth_id))	

	return (countries_dataset, records_dataset)

def get_parsed_dataset():
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

		new_records.append((name_array, expected_val))
		

	records = np.array(new_records)

	return records

##########################################################################
############################ Running the RNN #############################
##########################################################################

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def derivative_sigmoid_given_sigmoid_val(sigmoid_value):
	return sigmoid_value * (1 - sigmoid_value)

def softmax(x):
	exp = np.exp(x - np.max(x))
	return exp / np.sum(exp)

def derivative_softmax_given_softmax_val(softmax_value):
	s = softmax_value.reshape(-1, 1)
	return np.diagflat(s) - np.dot(s, s.T)

def get_binary_cross_entropy(hypothesis, expected_result):
	a = -expected_result
	b = np.log(hypothesis + 1e-15)
	c = (1 - expected_result)
	d = np.log(1 - hypothesis + 1e-15)

	cost = np.multiply(a, b) - np.multiply(c, d)

	return np.sum(cost)

def main():
	records = get_parsed_dataset()

	# Select the records that have a country to 'China', 'UK', or 'Russia'
	records = np.array(list(filter(lambda x: x[1][45] == 1 or x[1][99] == 1, records)))	

	np.random.shuffle(records)

	# Use 70% for the training set and 30% for the validation set
	num_training_data = int(len(records) * 0.7)
	training_data, test_data = records[:num_training_data,:], records[num_training_data:,:]

	alpha = 0.1
	input_dimensions = 27
	hidden_dimensions = 497
	output_dimensions = 124
	epsilon_init = 0.12

	# This is a 27 x 200 matrix
	layer_1_weights = 2 * np.random.random((input_dimensions, hidden_dimensions)) * (2 * epsilon_init) - epsilon_init

	# This is a 200 x 124 matrix
	layer_2_weights = np.random.random((hidden_dimensions, output_dimensions)) * (2 * epsilon_init) - epsilon_init

	# This is a 200 x 200 matrix
	hidden_state_weights = 2 * np.random.random((hidden_dimensions, hidden_dimensions)) * (2 * epsilon_init) - epsilon_init

	# This is the bias for the first layer
	bias_1 = 1

	# This is the bias for the second layer
	bias_2 = 1

	# The number of epoches
	num_epoche = 300

	for _ in range(num_epoche):

		# Train on the dataset
		for record in training_data:
			name = record[0]
			num_chars = len(name)
			y = np.array([record[1]])
			
			# Stores the hidden state for each letter position.
			letter_pos_to_hidden_state = np.zeros((num_chars + 1, 1, hidden_dimensions))

			# Stores the layer 2 values for each letter position
			letter_pos_to_layer_2_values = np.zeros((num_chars, 1, hidden_dimensions))

			# Stores the hypothesis for each letter position
			letter_pos_to_hypothesis = np.zeros((num_chars, 1, output_dimensions))

			# The hidden state for the first letter position is all 0s.
			letter_pos_to_hidden_state[0] = np.zeros((1, hidden_dimensions))

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
				hypothesis = softmax(np.dot(layer_2_values, layer_2_weights) + bias_2)

				# Update the dictionaries
				letter_pos_to_layer_2_values[i] = layer_2_values
				letter_pos_to_hypothesis[i] = hypothesis
				np.append(letter_pos_to_hidden_state, layer_2_values)

				loss = get_binary_cross_entropy(hypothesis, y)
				letter_pos_to_loss[i] = loss
				overall_error += loss

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
				sigma_2 = np.multiply(np.dot(layer_2_weights, sigma_3), derivative_sigmoid_given_sigmoid_val(layer_2_values).T)

				delta_2 += np.dot(sigma_3, layer_2_values).T
				delta_h += np.dot(sigma_2, hidden_state)
				delta_1 += np.dot(sigma_2, X).T

			layer_1_weights -= alpha * delta_1
			layer_2_weights -= alpha * delta_2
			hidden_state_weights -= alpha * delta_h

		# Perform one hypothesis test on a random dataset
		random_record = test_data[random.randint(0, len(test_data) - 1)]

		name = random_record[0]
		num_chars = len(name)
		y = np.array([random_record[1]])
		
		# Stores the hidden state for each letter position.
		letter_pos_to_hidden_state = np.zeros((num_chars + 1, 1, hidden_dimensions))

		# Stores the layer 2 values for each letter position
		letter_pos_to_layer_2_values = np.zeros((num_chars, 1, hidden_dimensions))

		# Stores the hypothesis for each letter position
		letter_pos_to_hypothesis = np.zeros((num_chars, 1, output_dimensions))

		# The hidden state for the first letter position is all 0s.
		letter_pos_to_hidden_state[0] = np.zeros((1, hidden_dimensions))

		overall_error = 0

		# Perform forward propagation
		for i in range(num_chars):
			
			# The inputs
			letter_array = name[i]
			X = np.array([letter_array]) 
			hidden_state = letter_pos_to_hidden_state[i - 1]

			# Perform forward propagation
			layer_2_values = sigmoid(np.dot(X, layer_1_weights) + np.dot(hidden_state, hidden_state_weights) + bias_1)
			hypothesis = softmax(np.dot(layer_2_values, layer_2_weights) + bias_2)

			# Update the dictionaries
			letter_pos_to_layer_2_values[i] = layer_2_values
			letter_pos_to_hypothesis[i] = hypothesis
			np.append(letter_pos_to_hidden_state, layer_2_values)

			loss = get_binary_cross_entropy(hypothesis, y)
			overall_error += loss

		print('hypothesis[-1]:', hypothesis[-1])
		print('y:', y)
		print('overall_error:', overall_error)
main()


