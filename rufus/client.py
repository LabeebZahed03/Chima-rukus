import json

from .scraper.crawler import Crawler
from .scraper.parser import Parser

class RufusClient:
    def __init__(self):
        self.crawler = Crawler()
        self.parser = Parser()

    def scrape(self, url: str, instructions: str, depth: int = 2):
        # Crawl the website
        raw_data = self.crawler.crawl(url, depth=depth, base_url=url)
        structured_data = []

        for page_url, (html, is_xml) in raw_data.items():
            parsed_data = self.parser.parse(html, instructions, is_xml=is_xml)
            if parsed_data:
                structured_data.append({
                    'url': page_url,
                    'data': parsed_data
                })

        return structured_data