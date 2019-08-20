import copy, numpy as np

def sigmoid(x):
	output = 1 / (1 + np.exp(-x))
	return output

def derivative_of_sigmoid(output):
	return output * (1 - output)

# Maps an integer to its binary form.
# Ex: int_to_binary[255] -> array([1, 1, 1 1, 1, 1, 1, 1])
int_to_binary = {}

binary_dim = 8
largest_number = pow(2, binary_dim)

# This stores the binary numbers as a list
# Ex: binary[254] -> [1, 1, 1, 1, 1, 1, 1, 0]
binary = np.unpackbits(
	np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
	int_to_binary[i] = binary[i]

# The input variables
alpha = 0.1
input_dimensions = 2
hidden_dimensions = 16
output_dimensions = 1

# We initialize the neural network weights to random values
synapse_0 = 2 * np.random.random((input_dimensions, hidden_dimensions)) - 1
synapse_1 = 2 * np.random.random((hidden_dimensions, output_dimensions)) - 1
synapse_h = 2 * np.random.random((hidden_dimensions, hidden_dimensions)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

print(synapse_1)
print(synapse_1_update)

# The training logic:
for j in range(100000):

	# Get a random addition 
	a_int = np.random.randint(largest_number / 2)
	a_binary = int_to_binary[a_int]

	b_int = np.random.randint(largest_number / 2)
	b_binary = int_to_binary[b_int]

	c_int = a_int + b_int
	c_binary = int_to_binary[c_int]

	# This is going to store what the neural network is going to predict from
	# a_binary + b_binary
	d_binary = np.zeros_like(c_binary)

	overall_error = 0
	layer_2_deltas = []
	layer_1_values = []
	layer_1_values.append(np.zeros(hidden_dimensions))

	print(a_int, a_binary)
	print(b_int, b_binary)
	input("Press Enter to continue...")

	# Now we iterate from the 1st binary position to the next binary position
	for binary_position in range(binary_dim):
		print("For position", binary_position)

		a_bit = a_binary[binary_dim - binary_position - 1]
		b_bit = b_binary[binary_dim - binary_position - 1]
		c_bit = c_binary[binary_dim - binary_position - 1]

		print('a_bit:', a_bit)
		print('b_bit:', b_bit)
		print('c_bit:', c_bit)

		input("Press Enter to continue...")

		X = np.array([[a_bit, b_bit]])
		y = np.array([[c_bit]]).T

		print('X:', X)
		print('y:', y)

		input("Press Enter to continue...")

		# Now we compute the hidden layer values
		print('synapse_0:', synapse_0)
		print('layer_1_values:', layer_1_values)
		print('synapse_h:', synapse_h)
		layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
		print('layer_1:', layer_1)
		input("Press Enter to continue...")

		# Now we compute the output layers
		print("synapse_1:", synapse_1)
		layer_2 = sigmoid(np.dot(layer_1, synapse_1))
		print("layer_2:", layer_2)
		input("Press Enter to continue...")

		# Now we compute the error
		layer_2_error = y - layer_2
		layer_2_deltas.append((layer_2_error) * derivative_of_sigmoid(layer_2))
		overall_error += np.abs(layer_2_error[0])

		# We now save the value we obtained into d_binary
		d_binary[binary_dim - binary_position - 1] = np.round(layer_2[0][0])

		# Store the hidden layer so that we can use it in the next timestamp
		layer_1_values.append(copy.deepcopy(layer_1))

	future_layer_1_delta = np.zeros(hidden_dimensions)

	synapse_1_update = 0
	synapse_h_update = 0
	synapse_0_update = 0

	# Now we need to back propagate
	for position in range(binary_dim):
		X = np.array([[a_binary[position], b_binary[position]]])
		layer_1 = layer_1_values[-position - 1]
		prev_layer_1 = layer_1_values[-position - 2]

		# Now we calculate the error in the output layer
		layer_2_delta = layer_2_deltas[-position - 1]

		# Now we calculate the error in the hidden layer
		layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * derivative_of_sigmoid(layer_1)

		# Now we update the weights so we can try again
		synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
		synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
		synapse_0_update += X.T.dot(layer_1_delta)

		future_layer_1_delta = layer_1_delta

	# Now we update the synapses
	synapse_0 += synapse_0_update * alpha
	synapse_1 += synapse_1_update * alpha
	synapse_h += synapse_h_update * alpha

	# We now print out the progress
	if j % 1000 == 0:
		print("Error:", overall_error)
		print("Predicted Value:", d_binary)
		print("Actual Value:", c_binary)
		
		d_int = 0
		for index, x in enumerate(reversed(d_binary)):
			d_int += x * pow(2, index)

		print(a_int, "+", b_int, "=", d_int)
		print("======================")



