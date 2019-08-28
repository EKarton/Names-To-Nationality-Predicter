import numpy as np
from NamesToNationalityClassifier import NamesToNationalityClassifier

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

'''
    It will return three values:
    1. A list of all possible labels
    2. A list of examples
    3. A list of labels where label[i] is the label for example[i]
'''
def get_dataset():
    country_id_to_country = get_countries()
    countries = [ country_id_to_country[id][0] for id in country_id_to_country ]

    records = [( record[0], country_id_to_country[record[1]][0] ) for record in get_records()]
    # records = list(filter(lambda x: x[1] == 'China' or x[1] == 'United Kingdom', records))
    np.random.shuffle(records)

    examples = [ record[0] for record in records ]
    labels = [ record[1] for record in records ]

    return countries, examples, labels

def main():
    countries, examples, labels = get_dataset()

    classifier = NamesToNationalityClassifier(examples, labels, countries)

    # Train the model
    classifier.train()
    classifier.save_model('data')

    # Make predictions
    # classifier.load_model_from_file('data.npz')
    # print(classifier.predict('Lon Shee'))

main()
