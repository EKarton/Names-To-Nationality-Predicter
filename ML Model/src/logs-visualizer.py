import matplotlib.pyplot as plt 

file_name = "results/Test-19-logs.csv"

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
    plt.subplot(2, 2, 1)
    plt.scatter(epoches, test_avg_errors)
    plt.title('Test Avg. Error vs Epoche')

    # Plot the test_accuracy vs epoche
    plt.subplot(2, 2, 2)
    plt.scatter(epoches, test_accuracies)
    plt.title('Test Accuracy vs Epoche')

    # Plot the train_avg_error vs epoche
    plt.subplot(2, 2, 3)
    plt.scatter(epoches, train_avg_errors)
    plt.title('Train Avg. Error vs Epoche')

    # Plot the train_accuracy vs epoche
    plt.subplot(2, 2, 4)
    plt.scatter(epoches, train_accuracies)
    plt.title('Train Accuracy vs Epoche')

    plt.show()

main()