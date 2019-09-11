# Names To Nationality Predicter - Web Scrapper

### Description
This module is responsible for scrapping names and their nationalities from Ancestry.com. The names and their nationalities are to be used for training the machine learning model.

### Table of Contents
- Overview
- Installation
- Running the App
- Credits
- License

### Overview
This module is comprised of various components:
1. The master process
2. The worker process(es)

The master process is responsible for:
- Creating the database schemas
- Adding the countries to the database
- Adding jobs (i.e, names to be scrapped who were born and have died at a particular country) to the Job Queue

The worker process is responsible for:
- Picking up any task in the Job Queue, scrape the data, and populate the data to the database

The job queue is managed by Redis, and the database of choice is Postgres.

Below is a diagram of the system architecture:
<div width="100%">
    <p align="center">
<img src="https://raw.githubusercontent.com/EKarton/Names-To-Nationality-Predictor/master/Web%20Scrapper/docs/System%20Architecture.png" width="600px"/>
    </p>
</div>

### Installation

Note that for the installation process to go through smoothly, ensure that you have:
1. Python 3
2. Unix machine (Mac, Linux, etc)

After ensuring the prerequisites are met above, follow the instructions below in order:
1. Install and configure Postgresql (skip this if you already have Postgresql):
	a. First, download a copy of the Postgresql Server on your machine, by typing the command:
		```sudo apt-get install ```

	b. Next, create a new user named ```webscraper``` by running the command:
		```wasd```

2. Install and configure the Redis Job Queue (skip this if you already have Redis):
	a. First, download a copy of redis on your machine, by typing the command:
		```installation```

	b. Next, keep note of where the redis server and the redis-cli lies, by typing the command:
		```which redis-server```
		and
		```which redis-cli```

	c. Then, create a new service file in ```/etc/systemd/system/redis.service``` with the contents:
		```
			[Unit]
			Description=Redis In-Memory Data Store
			After=network.target

			[Service]
			User=redis
			Group=redis
			ExecStart=REDIS_SERVER_PATH /etc/redis/redis.conf --supervised systemd
			ExecStop=REDIS_CLI_PATH shutdown
			Restart=always
			Type=notify

			[Install]
			WantedBy=multi-user.target
		```

		where REDIS_SERVER_PATH and REDIS_CLI_PATH are the values you got from step 2b.

	d. In addition, create a redis user by typing the command:
		```sudo adduser --system --group --no-create-home redis```

	e. Moreover, create the ```/var/lib/redis``` directory by running the command:
		```sudo mkdir /var/lib/redis```

	f. Now, give the redis user and group ownership over this directory:
		```sudo chown redis:redis /var/lib/redis```

		and make it accessible only to the redis user:
		```sudo chmod 770 /var/lib/redis```


	d. In addition, reload the systemctl daemon by typing the command:
		```sudo systemctl daemon-reload```

	e. Finally, start the Redis server by running the command:
		```sudo systemctl start redis```

### Running the app:
After completing the installation steps above, follow the steps below to get your app running:
1. First, ensure that Redis and Postgresql servers are online
2. Next, run the ```master.py``` script by running the command:
	```python3 master.py```

3. Once the script above is complete, run the ```worker.py``` script by running the command:
	```python3 worker.py```

* Note that multple workers can be run at the same time to make web scraping much faster. Just run more ```worker.py``` processes concurrently on the same machine or on different machines(*).

* Note that in order for the execution of different machines to work, the Redis and Postgresql servers need to be in a remote location that could be accessible by the  machines.


Postgresql tips:
- To install Postgresql Server on Ubuntu:
	https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-18-04

- To start the server:
	```sudo systemctl enable postgresql```
	```sudo systemctl start postgresql```

- To stop the server:
	```sudo systemctl disable postgresql```
	```sudo systemctl stop postgresql```

- To see if the server is running or not:
	```sudo systemctl status postgresql```
	```ps -aux | grep postgres```

- To install the adapter for Python:
	```sudo apt-get install python-psycopg2```
	```sudo apt-get install libpq-dev```
	```pip3 install psycopg2```

Redis job queue tips:
- To install Redis on Ubuntu:
	https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04

	Also, create a new service file in ```/etc/systemd/system/redis.service``` with the contents:
	```
		[Unit]
		Description=Redis In-Memory Data Store
		After=network.target

		[Service]
		User=redis
		Group=redis
		ExecStart=/usr/bin/redis-server /etc/redis/redis.conf --supervised systemd
		ExecStop=/usr/bin/redis-cli shutdown
		Restart=always
		Type=notify

		[Install]
		WantedBy=multi-user.target
	```

- To start the server:
	```sudo systemctl start redis```

- To stop the server:
	```sudo systemctl stop redis```

- To start the server at boot time:
	```sudo systemctl enable redis```

- To stop the server from starting at boot time:
	```sudo systemctl disable redis```

- To see if the server is running or not:
	```sudo systemctl status redis```
	```sudo service redis status```
	```ps -aux | grep redis```

- To see jobs:
	```npm install -g redis-commander```
	```redis-commander```

	On your browser, go to ```localhost:8081```

To see current jobs in Redis:
- You can see the current jobs in Redis visually by using ```redis-commander```:
1. Note that ```redis-commander``` is a NPM package, so to get it, run the command:
	```npm install -g redis-commander```

2. To run the NPM package, run the command:
	```./redis-commander```

3. Finally, to see all current jobs, go to the browser and navigate to ```localhost:8081```.

### Credits
Emilio Kartono, the sole creator of ths app

### License
Please note that this project is used for educational purposes and is not to be used commercially. We are not liable for any damages or changes done by this project.
Please refer to LICENCE.txt in the root project directory for further details.