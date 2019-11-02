# Names-To-Nationality-Predictor

### Description
Get the nationality from your name! This project attempts the predict the nationality of a person's name using recurrent machine learning model. A web app was built to make the machine learning model user friendly

### Table of Contents
- Walkthrough
- Results
- Installation
- Usage
- Credits
- License

### Walkthrough of the Project
A web app was developed to present the project in a more visually, interactive way:

### Results:
Using RNNs without any use of libraries posed plenty of challenges. First, utilizing only Numpy and Python to construct the RNN had performance issues. Regardless, the accuracy of the model remained to be decent - 86% accuracy in classifying a name to either Japanese, Vietnamese, Chinese, or Korean. Unfortunately, as the number of nationalities increases, accuracy suffers where only a 10% accuracy rate was obtained to classify a name to either one of the 124 countries worldwide. This makes sense since the world is now globalized and we cannot accurately predict people's nationalities solely on their names.

A report on how hyper-parameters were chosen are listed at: 

### Installation

#### Running the App with Docker
- Using Docker is a great way to run the app without having to deal with any system configurations
- By using Docker, it will use the pre-trained model found in the ```ML Model/data``` folder
- These are the instructions to get started with Docker:
	1. Ensure that you have docker installed
	2. Open up the terminal and change directories to the root project directory
	3. Build the docker image with the command:
		```docker build -t names-to-nationality .```
	4. Run the docker image with the command:
		```docker run -p 5000:5000 names-to-nationality```
	5. Open up your web browser and navigate to ```localhost:5000```

### Usage

### Credits

### License
