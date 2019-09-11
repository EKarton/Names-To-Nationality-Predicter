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

from urllib.parse import urlparse, urlencode, parse_qs

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

class ResultsPageWithRequests:
    def __init__(self, driver, country):
        self.driver = driver
        self.country = country

        # Wait for the browser to go to the new URL
        element = self.driver.find_element_by_xpath('//*[@id="searchPageSize"]')
        print('Waiting to land on search results page', element is not None)

        # Parse the current URL
        cur_url = self.driver.current_url
        parsed_url = urlparse(cur_url)
        query_params_string = parsed_url.query
        query_params = dict(parse_qs(query_params_string))

        # Store what is needed for a request
        self.__pg__ = 1
        self.__birth__ = query_params['birth']
        self.__birth_x__ = query_params['birth_x']
        self.__count__ = 50

        # Create a request session
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def set_num_results_per_page(self, value):
        self.__count__ = int(value)

    def goto_next_page(self):
        self.__pg__ += 1

    def get_current_page_number(self):
        return self.__pg__

    def get_max_page_number(self):

        # Make the first request to get the max number of records
        result = self.session.get(self.__create_api_url__())
        json_result = result.json()

        # Get the number of records available
        max_records = json_result['results']['hitCount']

        return max_records // self.__count__

    def get_names(self):
        names = []

        request_results = requests.get(self.__create_api_url__()).json()
        search_results = request_results['results']['items']

        for search_result in search_results:
            fields = search_result['fields']

            is_valid_name = False
            name = None
            cur_field_index = 0
            while name is None and cur_field_index < len(fields):
                cur_field = fields[cur_field_index]

                if cur_field['label'] == 'Name':
                    name = cur_field['text']

                if cur_field['label'] == 'Birth' and self.country in cur_field['text'].lower():
                    is_valid_name = True

                cur_field_index += 1

            if name is not None and is_valid_name:
                names.append(name)

        return names

    def __create_api_url__(self):
        # Put all the URL parameters into a hashmap
        url_params = {}
        url_params['birth'] = self.__birth__
        url_params['birth_x'] = self.__birth_x__
        url_params['count'] = self.__count__
        url_params['pg'] = self.__pg__
        url_params['searchState'] = 'Pagination'
        url_params['useClusterView'] = 'useClusterView'
        
        # Create the URL with the URL params
        base_url = 'https://www.ancestry.com/api/search-results?'
        new_url = base_url + urlencode(url_params)
        return new_url

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

            results_page = ResultsPageWithRequests(browser, country)
            results_page.set_num_results_per_page('50')

            max_page_number = results_page.get_max_page_number()
            num_records_obtained = 0
            cur_num_records_in_db = self.records_database.get_num_records(country_id)
            max_records = min(max_page_number * 50, max_records - cur_num_records_in_db)

            print("Already have", cur_num_records_in_db, "records born on", country, " in the database")
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

            print("Completed getting records for country", country)

        except Exception as error:
            print("ERROR AT", time.ctime())
            browser.quit()
            raise error

        finally:
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
