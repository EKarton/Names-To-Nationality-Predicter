### Configuring Postgresql:
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

### Configuring the Redis job queue:
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

