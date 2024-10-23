import os
import json
from datetime import datetime

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

    def scrape(self, url: str, instructions: str, depth: int = 2, output_file: str = None):
        raw_data = self.crawler.crawl(url, depth=depth, base_url=url)
        structured_data = []

        for page_url, (html, is_xml) in raw_data.items():
            parsed_data = self.parser.parse(html, instructions, url=page_url, is_xml=is_xml)
            if parsed_data:
                structured_data.append(parsed_data)

        # Save data into file
        if output_file is None:
            # Set default output folder and file with timestamp
            output_folder = 'outputs'
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_folder, f'documents_{timestamp}.json')
        else:
            output_folder = os.path.dirname(output_file)
            if output_folder and not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=4, ensure_ascii=False)

            print(f"Data saved to {output_file}")
        except Exception as e:
            print(f"Failed to save data to {output_file}: {e}")
            raise

        return structured_data
