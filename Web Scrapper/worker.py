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

from redis_queue import RedisQueue
import json

from database import RecordsDB
from database import CountriesDB

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

class RecordsParser:
	def __init__(self, records_database):
		self.records_database = records_database

	def get_records(self, country, country_id, max_records=sys.maxsize):
		try:
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
				new_records_saved = 0
				for name in names_on_page:
					if not self.records_database.has_record(name, country_id):
						self.records_database.add_record(name, country_id)
						new_records_saved += 1
				print(new_records_saved, 'names saved to database')

				num_records_obtained += new_records_saved
				results_page.goto_next_page()
				print('Progress:', str((num_records_obtained / max_records) * 100) + '% (Collected', num_records_obtained, 'out of', str(max_records) + ')')

		except Exception as error:
			print("ERROR AT", time.ctime())
			print(error)

		finally:
			self.records_database.shutdown()
			browser.quit()	

def main():
	with RecordsDB() as records_db:
		records_parser = RecordsParser(records_db)

		with CountriesDB() as countries_db:
			queue = RedisQueue(name='jobs', namespace='queue', decode_responses=True)
			job_in_json = queue.wait_and_dequeue()

			while job_in_json is not None:

				job = json.loads(job_in_json)

				country_id = job['country_id']
				country_name = countries_db.get_country_from_id(country_id)
				num_records = job['num_records']

				if country_name is None:
					raise Exception("Country name cannot be None!")

				records_parser.get_records(country=country_name, country_id=country_id, max_records=num_records)

				job_in_json = queue.wait_and_dequeue()

if __name__ == "__main__":
    main()