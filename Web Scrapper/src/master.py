from database import CountriesDB
from database import RecordsDB
from redis_queue import RedisQueue
import json

'''
	The purpose of this script is to initialize the postgresql database with our data model.
'''

def initialize_records_db():
	with RecordsDB() as records_db:
		records_db.create_table_if_not_exist()
		records_db.clear_all_data()


def save_countries_to_db():

	with CountriesDB() as countries_db:
		with open('country-to-nationality.txt') as reader:

			countries_db.create_table_if_not_exist()
			countries_db.clear_all_data()

			# Skip the first line (since first line is just the csv header)
			reader.readline()

			line = reader.readline()
			while line:
				print(line)
				tokenized_lines = line.split(',')
				if len(tokenized_lines) >= 4:
					country = tokenized_lines[1]
					nationality = tokenized_lines[3]

					countries_db.insert_country(country, nationality)

				line = reader.readline()

def populate_job_queue():
	queue = RedisQueue('jobs')
	with CountriesDB() as countries_db:

		countries = countries_db.get_countries()
		for country in countries:
			job = {'country_id': country[0], 'num_records': 5000}
			job_in_json = json.dumps(job)
			queue.enqueue(job_in_json)

# initialize_records_db()
# save_countries_to_db()
populate_job_queue()

# queue = RedisQueue('jobs')
# job = {'country_id': 5978, 'num_records': 500}
# job_in_json = json.dumps(job)
# queue.enqueue(job_in_json)

# job = {'country_id': 5998, 'num_records': 500}
# job_in_json = json.dumps(job)
# queue.enqueue(job_in_json)