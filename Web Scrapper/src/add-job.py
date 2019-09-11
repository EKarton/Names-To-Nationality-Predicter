from database import CountriesDB
from database import RecordsDB
from redis_queue import RedisQueue
import json

import argparse

def add_job_to_queue(country_id, num_records):
	queue = RedisQueue('jobs')
	job = {'country_id': country_id, 'num_records': num_records}
	job_in_json = json.dumps(job)
	queue.enqueue(job_in_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add job(s) to the queue')

    parser.add_argument('-c', '--country_id', nargs='+', metavar='Country ID', help='List of country IDs to add to the queue')
    parser.add_argument('-n', '--num-records', nargs=1, metavar='Num Records', help="Expected number of records to finish with")

    args = parser.parse_args()

    country_ids = args.country_id
    num_records = args.num_records

    for country_id in country_ids:
        add_job_to_queue(country_id, num_records)
