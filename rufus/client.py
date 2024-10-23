import json

from .scraper.crawler import Crawler
from .scraper.parser import Parser


class RufusClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self._validate_api_key():
            raise ValueError("Invalid API key.")
        self.crawler = Crawler()
        self.parser = Parser()

    def _validate_api_key(self):
        # Placeholder for API key validation logic
        # Currently accepts any non-empty string
        return bool(self.api_key)

    def scrape(self, url: str, instructions: str, depth: int = 2):
        
        raw_data = self.crawler.crawl(url, depth=depth, base_url=url)
        structured_data = []

        for page_url, (html, is_xml) in raw_data.items():
            parsed_data = self.parser.parse(html, instructions, url=page_url, is_xml=is_xml)
            if parsed_data:
                structured_data.append(parsed_data)

        return structured_data