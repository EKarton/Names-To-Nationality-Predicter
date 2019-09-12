import numpy as np
from names_to_nationality_classifier import NamesToNationalityClassifier

'''
    Obtains a map from country ID to country name.
    For example,
    {
        5998: "United Kingdom",
        5978: "China",
        ...
    }
'''
def get_countries():
	country_id_to_country_name = {}

	with open('data/countries.csv') as countries_file_reader:

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

'''
    Obtains the records from the CSV file into a list.
    For example,
    [
        ("Bob Smith", 5998),
        ("Xi Jinping", 5978),
        ...
    ]
'''
def get_records():
	records = []
	with open('data/records.csv') as reader:

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
    1.  A list of all possible labels
        For example,
        [
            "United Kingdom", 
            "China", 
            ...
        ]

    2.  A list of examples
        For example,
        [
            "Bob Smith",
            "Xi Jinping",
            ...
        ]

    3.  A list of labels where label[i] is the label for example[i]
        For example,
        [
            "United Kingdom",
            "China",
            ...
        ]

    It returns in the order listed above
'''
def get_dataset():
    country_id_to_country = get_countries()
    countries = [ country_id_to_country[id][0] for id in country_id_to_country ]
	
    records = [( record[0], country_id_to_country[record[1]][0] ) for record in get_records()]
    # records = list(filter(lambda x: x[1] == 'China' or x[1] == 'United Kingdom', records))
    # countries = ["China", "United Kingdom"]
    np.random.shuffle(records)
	
    # Splits the records into two lists
    examples = [ record[0] for record in records ]
    labels = [ record[1] for record in records ]

    return countries, examples, labels

'''
    The main method
'''
def main():
    countries, examples, labels = get_dataset()

    classifier = NamesToNationalityClassifier(countries)
    classifier.add_training_examples(examples, labels)

    # Train the model
    try:
        print('Training data')
        classifier.train()
    finally:
        print('Saved model to data.npz')
        classifier.save_model('data/data')

    # Make predictions
    # classifier.load_model_from_file('data/data.npz')
    # print(classifier.predict('Raymond Zhang'))

main()
