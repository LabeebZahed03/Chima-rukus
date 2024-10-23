import json
from rufus.scraper.crawler import Crawler
from rufus.scraper.parser import Parser

class RufusClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.crawler = Crawler()
        self.parser = Parser()

    def scrape(self, url: str, instructions: str, depth: int = 2):
        # Crawl the website
        raw_data = self.crawler.crawl(url, depth=depth, base_url=url)
        structured_data = []

        for page_url, html in raw_data.items():
            is_xml = page_url.endswith('.xml')
            parsed_data = self.parser.parse(html, instructions, is_xml=is_xml)
            if parsed_data:
                structured_data.append({
                    'url': page_url,
                    'data': parsed_data
                })

        return structured_data

    def save_to_file(self, data, filename='output.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)