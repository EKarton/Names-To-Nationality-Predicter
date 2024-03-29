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
    print('Countries Filepath:', filepath)

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
def get_records(max_records_per_country=float("inf")):

    # We first put all the records from the file
    raw_records = []
    with open('data/records.csv') as reader:

        line = reader.readline()
        while line:
            tokenized_line = line.split(',')

            if len(tokenized_line) == 3:
                name = tokenized_line[1]
                country_of_birth_id = int(tokenized_line[2])
                raw_records.append((name, country_of_birth_id))

            line = reader.readline()

    # Shuffle the raw records to remove the potential ordering in the file
    np.random.shuffle(raw_records)

    # We then add the records to our dataset ensuring that it meets the count
    records = []
    country_id_to_num_records = {}
    for record in raw_records:
        country_of_birth_id = record[1]

        if country_of_birth_id not in country_id_to_num_records:
            records.append(record)
            country_id_to_num_records[country_of_birth_id] = 1

        elif country_id_to_num_records[country_of_birth_id] < max_records_per_country:
            records.append(record)
            country_id_to_num_records[country_of_birth_id] += 1

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

    Note: For data/china-korea-japan-vietnam-countries.csv, use the following hyper-params:
    - Momentum = 0.9
    - L2 = 0.0001
    - Learning Rate = 0.0001
    - Hidden Dimensions: 200
    - Epoche: 50

    Note: For data/countries-without-usa-or-canada.csv, use the following hyper-params:
    - Momentum = 0.9
    - L2 = 0
    - Learning Rate = 0.0001
    - Hidden Dimensions: 500
    - Epoche: 20
'''
def get_dataset():
    # country_id_to_country = get_countries(filepath='data/countries.csv')
    country_id_to_country = get_countries(filepath='data/countries-without-usa-or-canada.csv')
    # country_id_to_country = get_countries(filepath='data/china-korea-japan-vietnam-countries.csv')
    # country_id_to_country = get_countries(filepath='data/european-countries.csv')
    countries = [ country_id_to_country[id][0] for id in country_id_to_country ]
    countries.sort()

    records = get_records(max_records_per_country=5000)
    records = list(filter(lambda x: x[1] in country_id_to_country, records))
    records = [( record[0], country_id_to_country[record[1]][0] ) for record in records]

    # Shuffle the records
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
    plt.ioff()

    classifier = NamesToNationalityClassifier(countries, 
                                            alpha=0.0001,
                                            hidden_dimensions=500, 
                                            momentum=0.9,
                                            num_epoche=20,
                                            l2_lambda=0)

    classifier.add_training_examples(examples, labels)
    performance = classifier.train()

    epoches = [i for i in range(classifier.num_epoche)]

    # Plot the performance
    fig, (errors_plt, accuracy_plt) = plt.subplots(2)
    plt_title_format = "Performance. for Learning Rate: {:.5f}, Hidden Dim: {:.5f}, \nL2_lambda: {:.5f}, Momentum: {:.5f}, Num Epoche: {:.5f}"
    fig_title = plt_title_format.format(classifier.alpha, 
                                        classifier.hidden_dimensions, 
                                        classifier.l2_lambda, 
                                        classifier.momentum, 
                                        classifier.num_epoche)
    fig.suptitle(fig_title, fontsize=10)

    errors_plt.set_title('Errors vs Epoche', fontsize=10)
    errors_plt.plot(epoches, performance['epoche_to_train_avg_error'], label='Train Avg. Error')
    errors_plt.plot(epoches, performance['epoche_to_test_avg_error'], label='Test Avg. Error')
    errors_plt.legend()
    errors_plt.set_xlabel('Epoche')
    errors_plt.set_ylabel('Error')

    accuracy_plt.set_title('Accuracy vs Epoche', fontsize=10)
    accuracy_plt.plot(epoches, performance['epoche_to_train_accuracy'], label='Train Accuracy')
    accuracy_plt.plot(epoches, performance['epoche_to_test_accuracy'], label='Test Accuracy')
    accuracy_plt.legend()
    accuracy_plt.set_xlabel('Epoche')
    accuracy_plt.set_ylabel('Accuracy')

    plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(hspace=0.5)

    # Save the plot
    plt_file_name_format = 'L{}-H-{}-R-{}-M-{}-E-{}-plots.png'
    plt_file_name = plt_file_name_format.format(classifier.weight_init_type,
                                                str(classifier.hidden_dimensions).replace('.', '_'), 
                                                str(classifier.alpha).replace('.', '_'), 
                                                str(classifier.momentum).replace('.', '_'), 
                                                str(classifier.num_epoche).replace('.', '_'))
    plt.savefig(plt_file_name)

    # Save the data
    data_file_name_format = 'L{}-H-{}-R-{}-M-{}-E-{}-data'
    data_file_name = data_file_name_format.format(classifier.weight_init_type,
                                                    str(classifier.hidden_dimensions).replace('.', '_'), 
                                                    str(classifier.alpha).replace('.', '_'), 
                                                    str(classifier.momentum).replace('.', '_'), 
                                                    str(classifier.num_epoche).replace('.', '_'))
    print('Saved model to', data_file_name + '.npz')
    classifier.save_model('data/' + data_file_name)

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

if __name__ == "__main__":
    main()
