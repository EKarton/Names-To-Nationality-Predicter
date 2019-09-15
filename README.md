# Names-To-Nationality-Predictor
Predicts what nationality you are based on your name

### Running the App with Docker
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