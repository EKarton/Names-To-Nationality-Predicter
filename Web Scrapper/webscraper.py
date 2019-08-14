import sys
import psycopg2
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC

class SearchPage:
	def __init__(self, driver):
		self.driver = driver

	def goto_page(self):
		self.driver.get('https://www.ancestry.ca/search/')

	def show_more_options(self):
		self.driver.find_element_by_xpath('//*[@id="sfs_Global"]/div[3]/div[1]/button/span[1]').click()

	def add_birth_event(self):
		self.driver.find_element_by_xpath('//*[@id="sfs_addEventLink_SelfBirth"]').click()

	def type_country_of_birth(self, country):
		self.driver.find_element_by_xpath('//*[@id="sfs_LifeEventSelectormsbpn__ftp"]').send_keys(country)

	def select_exact_to_country(self):
		self.driver.find_element_by_xpath('//*[@id="sfs_Global"]/div[3]/div[2]/div[1]/div/div[2]/fieldset/div[7]/div/button/span').click()

	def submit(self):
		self.driver.find_element_by_xpath('//*[@id="sfs_Global"]/div[3]/div[2]/div[7]/input').click()

class ResultsPage:
	def __init__(self, driver):
		self.driver = driver

	def set_num_results_per_page(self, value):
		element =  self.driver.find_element_by_xpath('//*[@id="searchPageSize"]')
		Select(element).select_by_visible_text(value)

	def goto_next_page(self):
		next_button = self.driver.find_element_by_id('pagingInfo').find_elements_by_tag_name('a')[1]
		next_button.click()
		WebDriverWait(self.driver, 20).until(EC.invisibility_of_element_located((By.ID, 'searchResultActivityIndicator')))

	def get_current_page_number(self):
		select_element = Select(self.driver.find_element_by_xpath('//*[@id="selectSomething"]'))
		return int(select_element.first_selected_option.text)

	def get_max_page_number(self):
		raw_text = self.driver.find_element_by_xpath('//*[@id="pagingInfo"]/span[2]/span').text
		return int(raw_text.replace(',', ''))

	def get_names(self):
		names = []
		searchResults = self.driver.find_elements_by_class_name("searchResultRow")
		for searchResult in searchResults:
			name = searchResult.find_element_by_tag_name('tbody').find_elements_by_tag_name('tr')[0].find_element_by_tag_name('td').text
			names.append(name)

		return names

class RecordsDatabase:
	def __init__(self):
		self.connection = None
		self.cursor = None

	def start(self):
		print("Starting connection to database")
		try:
			self.connection = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="Uranium1122")
			self.cursor = self.connection.cursor()

		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
		print("Connected to database")

	def add_record(self, name, country_of_birth):
		sql = """INSERT INTO records (name, country_of_birth) 
				 VALUES(%s, %s);"""

		self.cursor.execute(sql, (name, country_of_birth))
		self.connection.commit()

	def shutdown(self):
		print("Stopping connection to database")

		if self.cursor is not None:
			self.cursor.close()

		print("Stopped connection to database")

class RecordsParser:
	def __init__(self, records_database):
		self.records_database = records_database

	def get_records(self, country, max_records=sys.maxsize):
		try:
			self.records_database = RecordsDatabase()
			self.records_database.start()

			chrome_options = Options()  
			chrome_options.add_argument("--headless")  

			browser = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
			browser.implicitly_wait(10)

			search_page = SearchPage(browser)
			search_page.goto_page()
			search_page.show_more_options()
			search_page.add_birth_event()
			search_page.type_country_of_birth(country)
			search_page.select_exact_to_country()
			search_page.submit()

			results_page = ResultsPage(browser)
			results_page.set_num_results_per_page('50')

			max_page_number = results_page.get_max_page_number()
			num_records_obtained = 0
			max_records = min(max_page_number * 50, max_records)

			print("Getting", max_records, "records born on", country)

			while results_page.get_current_page_number() < max_page_number and num_records_obtained < max_records:

				print('Page', results_page.get_current_page_number(), 'of', max_page_number)

				names_on_page = results_page.get_names()
				print('Obtained', len(names_on_page), 'names')

				print('Saving names to database')
				for name in names_on_page:
					self.records_database.add_record(name, country)
				print('Names saved to database')

				num_records_obtained += len(names_on_page)
				results_page.goto_next_page()
				print('Progress:', str((num_records_obtained / max_records) * 100) + '% (Collected', num_records_obtained, 'out of', str(max_records) + ')')

		except Exception as error:
			print("ERROR AT", time.ctime())
			print(error)

		finally:
			self.records_database.shutdown()
			browser.quit()	

def main():
	commandline_args = sys.argv[1:]

	if len(commandline_args) < 1:
		raise Exception("Needs at least one command line argument")

	try:
		records_database = RecordsDatabase()
		records_parser = RecordsParser(records_database)

		if len(commandline_args) == 1:
			records_parser.get_records(country=commandline_args[0])
		else:
			records_parser.get_records(country=commandline_args[0], max_records=commandline_args[1])

	except (Exception) as error:
		print("ERROR AT:", time.ctime())
		print(error)

if __name__ == "__main__":
    main()