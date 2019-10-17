import numpy as np
from names_to_nationality_classifier import NamesToNationalityClassifier
from collections import OrderedDict

# Make matplotlib not interactive
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt 

'''
    Obtains a map from country ID to country name.
    For example,
    {
        5998: ("United Kingdom", "British"),
        5978: ("China", "Chinese"),
        ...
    }
'''
def get_countries(filepath='data/countries.csv'):
	country_id_to_country_name = {}

	with open(filepath) as countries_file_reader:

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
    country_id_to_country = get_countries(filepath='data/china-korea-japan-countries.csv')
    countries = [ country_id_to_country[id][0] for id in country_id_to_country ]
    countries.sort()

    records = get_records()
    records = list(filter(lambda x: x[1] in country_id_to_country, records))
    records = [( record[0], country_id_to_country[record[1]][0] ) for record in records]
    # records = list(filter(lambda x: x[1] == 'China' or x[1] == 'United Kingdom', records))
    # countries = ["China", "United Kingdom"]
    
    np.random.shuffle(records)
    records = records[0:2000]

    print(records[0])
	
    # Splits the records into two lists
    examples = [ record[0] for record in records ]
    labels = [ record[1] for record in records ]

    return countries, examples, labels

'''
    The main method
'''
def main():
    countries, examples, labels = get_dataset()
    plt.ioff()

    # Test out different hyperparameters
    plt_title_format = "Perf. for Learning Rate: {:.5f}, Hidden Dim: {:.5f}, \nL2_lambda: {:.5f}, Momentum: {:.5f}, Num Epoche: {:.5f}"
    various_hidden_layers_count = [300]

    for hidden_layers_count in various_hidden_layers_count:
        classifier = NamesToNationalityClassifier(countries, hidden_dimensions=hidden_layers_count, momentum=0.1, num_epoche=100)
        classifier.add_training_examples(examples, labels)
        performance = classifier.train()

        epoches = [i for i in range(classifier.num_epoche)]

        # Plot the performance
        fig, (errors_plt, accuracy_plt) = plt.subplots(2)
        fig_title = plt_title_format.format(classifier.alpha, 
                                            classifier.hidden_dimensions, 
                                            classifier.l2_lambda, 
                                            classifier.momentum, 
                                            classifier.num_epoche)
        fig.suptitle(fig_title, fontsize=10)

        errors_plt.title.set_text('Errors vs Epoche')
        errors_plt.plot(epoches, performance['epoche_to_train_avg_error'], label='Train Avg. Error')
        errors_plt.plot(epoches, performance['epoche_to_test_avg_error'], label='Test Avg. Error')
        errors_plt.legend()

        accuracy_plt.title.set_text('Accuracy vs Epoche')
        accuracy_plt.plot(epoches, performance['epoche_to_train_accuracy'], label='Train Accuracy')
        accuracy_plt.plot(epoches, performance['epoche_to_test_accuracy'], label='Test Accuracy')
        accuracy_plt.legend()

        # Save the plot
        plt.savefig('L1-H' + str(hidden_layers_count).replace('.', '_') + '-R-' + str(classifier.alpha).replace('.', '_') + '-M-' + str(classifier.momentum).replace('.', '_') + '-E-' + str(classifier.num_epoche) + '-plots.png')

    # # Train the model
    # classifier = NamesToNationalityClassifier(countries)
    # try:
    #     print('Training data')
    #     classifier.add_training_examples(examples, labels)
    #     classifier.train()
    # finally:
    #     print('Saved model to data.npz')
    #     classifier.save_model('data/data')

    # Make predictions
    # classifier.load_model_from_file('data/data.npz')
    # print(classifier.predict('Emilio Kartono'))

main()
