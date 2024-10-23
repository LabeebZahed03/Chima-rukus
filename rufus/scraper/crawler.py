# rufus/scraper/crawler.py

import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # Import Service class
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from ..utils.logger import get_logger

class Crawler:
    def __init__(self):
        self.visited = set()
        self.logger = get_logger(__name__)

        #headless chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        
        service = Service(ChromeDriverManager().install())

        # Initialize the WebDriver with the Service and Options
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def crawl(self, url: str, depth: int = 2, base_url: str = None, retries: int = 3):
        if depth == 0 or url in self.visited:
            return {}

        self.visited.add(url)
        self.logger.info(f"Crawling: {url}")

        raw_data = {}

        for attempt in range(retries):
            try:
                self.driver.get(url)
                time.sleep(2)
                html = self.driver.page_source
                content_type = self.driver.execute_script("return document.contentType;")
                is_xml = 'xml' in content_type or url.endswith('.xml')
                break  # Successful fetch
            except WebDriverException as e:
                self.logger.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    backoff_time = 2 ** attempt
                    self.logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    self.logger.error(f"Failed to retrieve {url} after {retries} attempts.")
                    return {}

        raw_data[url] = (html, is_xml)

        if is_xml:
            return raw_data

        soup = BeautifulSoup(html, 'html.parser')
        links = set()

        for link in soup.find_all('a', href=True):
            href = link['href']
            parsed_href = urlparse(href)
            if parsed_href.scheme in ('http', 'https', ''):
                full_url = urljoin(base_url, href)
                if full_url.startswith(base_url):
                    links.add(full_url)

        for link in links:
            raw_data.update(self.crawl(link, depth=depth - 1, base_url=base_url))

        return raw_data

    def __del__(self):
        # Clean up driver when the crawler is destroyed
        self.driver.quit()
