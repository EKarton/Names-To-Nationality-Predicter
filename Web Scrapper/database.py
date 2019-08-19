import psycopg2
import traceback

class Database():
	def __init__(self, host='localhost', database_name='NamesToNationalityDB', user='webscraper', password='Uranium1122', table_name=''):
		self.connection = None
		self.host = host
		self.database_name = database_name
		self.user = user
		self.password = password
		self.table_name = table_name

	def start(self):
		print("Starting connection to database")
		self.connection = psycopg2.connect(host=self.host, database=self.database_name, user=self.user, password=self.password)
		print("Connected to database")

	def shutdown(self):
		print('Shutting down connection to server')
		if self.connection:
			self.connection.close()
		print('Shut down to server completed')

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, exc_type, exc_value, tb):
		self.shutdown()

	def clear_all_data(self):
		sql = 'DELETE FROM ' + self.table_name

		cursor = self.connection.cursor()
		cursor.execute(sql)
		cursor.close()

		self.connection.commit()

class CountriesDB(Database):

	def __init__(self, host='localhost', database_name='NamesToNationalityDB', user='webscraper', password='Uranium1122'):
		super().__init__(host, database_name, user, password, 'countries')

	def create_table_if_not_exist(self):
		sql = """CREATE TABLE IF NOT EXISTS countries(
					country_id INTEGER PRIMARY KEY,
					country VARCHAR(250) NOT NULL,
					nationality VARCHAR(250) NOT NULL
				);"""

		cursor = self.connection.cursor()
		cursor.execute(sql)
		cursor.close()

		self.connection.commit()

	def insert_country(self, country, nationality):
		sql = """INSERT INTO countries (country, nationality) 
				 VALUES(%s, %s);"""

		cursor = self.connection.cursor()
		cursor.execute(sql, (country, nationality))
		cursor.close()

		self.connection.commit()

	def get_countries(self):
		sql = """SELECT country_id, country, nationality 
				 FROM countries;"""

		cursor = self.connection.cursor()
		cursor.execute(sql)

		countries = []
		row = cursor.fetchone()
		while row is not None:
			countries.append(row)
			row = cursor.fetchone()

		cursor.close()

		return countries

	def get_country_from_id(self, country_id):
		sql = """SELECT country 
				 FROM countries 
				 WHERE country_id = %s;"""

		cursor = self.connection.cursor()
		cursor.execute(sql, (country_id, ))

		country_name = None
		row = cursor.fetchone()
		if row is not None and len(row) == 1:
			country_name = row[0]

		cursor.close()

		return country_name

class RecordsDB(Database):
	def __init__(self, host='localhost', database_name='NamesToNationalityDB', user='webscraper', password='Uranium1122'):
		super().__init__(host, database_name, user, password, 'records')

	def create_table_if_not_exist(self):
		sql = """CREATE TABLE IF NOT EXISTS records(
				 	record_id INTEGER PRIMARY KEY,
					name VARCHAR(250) NOT NULL,
					country_of_birth_id INTEGER REFERENCES records(country_id)
				);"""

		cursor = self.connection.cursor()
		cursor.execute(sql)
		cursor.close()

		self.connection.commit()

	def add_record(self, name, country_of_birth_id):
		sql = """INSERT INTO records (name, country_of_birth_id) 
				 VALUES(%s, %s);"""

		cursor = self.connection.cursor()
		cursor.execute(sql, (name, str(country_of_birth_id)))
		self.connection.commit()
		cursor.close()

	def has_record(self, name, country_of_birth_id):
		sql = """SELECT COUNT(record_id) 
				 FROM records 
				 WHERE name = %s AND country_of_birth_id = %s"""

		cursor = self.connection.cursor()
		cursor.execute(sql, (name, str(country_of_birth_id)))

		row = cursor.fetchone()
		cursor.close()
		return len(row) == 1 and row[0] > 0
