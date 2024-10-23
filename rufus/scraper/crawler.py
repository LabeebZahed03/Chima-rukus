import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self):
        self.visited = set()
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'RufusBot/1.0'
        }

    def crawl(self, url: str, depth: int = 2, base_url: str = None):
        if depth == 0 or url in self.visited:
            return {}

        self.visited.add(url)
        print(f"Crawling: {url}")

        try:
            response = self.session.get(url, headers=self.headers)
            content_type = response.headers.get('Content-Type', '')
            is_xml = 'xml' in content_type or url.endswith('.xml')
            html = response.text
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return {}

        raw_data = {url: (html, is_xml)}

        if is_xml:
            return raw_data  # Do not follow links in XML files

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