import copy, numpy as np

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

	print('countries:', countries)
	print('records:', records)

	alpha = 0.1
	input_dimensions = 27
	hidden_dimensions = 200
	output_dimensions = 124

	# This is a 27 x 200 matrix
	# Layer 1 represents the input layer
	layer_1_to_2_weights = 2 * np.random.random((input_dimensions, hidden_dimensions)) - 1 #np.random.rand(input_dimensions, hidden_dimensions)

	# This is a 200 x 124 matrix
	# Layer 2 represents the hidden layer
	layer_2_to_3_weights = 2 * np.random.random((hidden_dimensions, output_dimensions)) - 1 #np.random.rand(hidden_dimensions, output_dimensions)

	# This is a 200 x 200 matrix
	hidden_state_weights = 2 * np.random.random((hidden_dimensions, hidden_dimensions)) - 1 # np.random.rand(hidden_dimensions, hidden_dimensions)

	for record in records:
		name = record[0]
		expected_value = record[1]

		y = np.array([expected_value]).T 

		# We first initialize the hidden state to 0
		last_layer_2_values = np.zeros(hidden_dimensions)
		last_layer_3_values = None

		for letter_array in name:
			X = np.array([letter_array]) 

			# Perform forward propagation
			layer_2_values = sigmoid(np.dot(X, layer_1_to_2_weights) + np.dot(last_layer_2_values, hidden_state_weights))
			layer_3_values = sigmoid(np.dot(layer_2_values, layer_2_to_3_weights))

			# Update the last layer 1 and layer 2 values
			last_layer_2_values = layer_2_values
			last_layer_3_values = layer_3_values

			# Compute the error

		print('last_layer_3_values:', last_layer_3_values)
		input('Press enter to continue')
main()


