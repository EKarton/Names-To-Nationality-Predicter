import matplotlib.pyplot as plt 

file_name = "results/Testing.csv"

def main():
    epoches = []
    test_avg_errors = []
    test_accuracies = []
    train_avg_errors = []
    train_accuracies = []

    with open(file_name, 'r') as file_reader:
        line = file_reader.readline() # Skip the csv header

        line = file_reader.readline()
        while line:
            tokenized_line = line.split(',')

            epoche = int(tokenized_line[0])
            test_avg_error = float(tokenized_line[1])
            test_accuracy = float(tokenized_line[2])
            train_avg_error = float(tokenized_line[3])
            train_accuracy = float(tokenized_line[4])

            epoches.append(epoche)
            test_avg_errors.append(test_avg_error)
            test_accuracies.append(test_accuracy)
            train_avg_errors.append(train_avg_error)
            train_accuracies.append(train_accuracy)

            line = file_reader.readline()

    # Plot the test_avg_error vs epoche
    '''
        cross_entropies_plt.title.set_text('Cross entropies vs Epoche')
            cross_entropies_plt.plot(iterations, cross_entropies_train, label="Cross Entropies Train")
            cross_entropies_plt.plot(iterations, cross_entropies_valid, label="Cross Entropies Valid")
            cross_entropies_plt.legend()
    '''
    fig, (errors_plt, accuracy_plt) = plt.subplots(2)

    errors_plt.title.set_text('Errors vs Epoche')
    errors_plt.plot(epoches, train_avg_errors, label='Test Avg. Error')
    errors_plt.plot(epoches, test_avg_errors, label='Test Avg. Error')
    errors_plt.legend()

    accuracy_plt.title.set_text('Accuracy vs Epoche')
    accuracy_plt.plot(epoches, train_accuracies, label='Train Accuracy')
    accuracy_plt.plot(epoches, test_accuracies, label='Test Accuracy')
    accuracy_plt.legend()

    plt.show()

main()