import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self):
        self.session = requests.Session()
        self.visited = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

    def crawl(self, url: str, depth: int = 2, base_url: str = None):
        if depth < 1 or url in self.visited:
            return {}

        self.visited.add(url)
        print(f"Crawling: {url}")

        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}

        data = {url: html}

        # Extract and crawl links
        soup = BeautifulSoup(html, 'html.parser')
        links = self.extract_links(soup, url, base_url)

        for link in links:
            if link not in self.visited:
                data.update(self.crawl(link, depth - 1, base_url=base_url))

        return data

    def extract_links(self, soup: BeautifulSoup, current_url: str, base_url: str):
        links = set()
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            href = urljoin(current_url, href)
            if self.is_valid_url(href, base_url):
                links.add(href)
        return links

    def is_valid_url(self, url: str, base_url: str):
        parsed = urlparse(url)
        if not bool(parsed.netloc) and not bool(parsed.scheme):
            return False
        # Ensure the link is within the same domain
        if base_url and urlparse(url).netloc != urlparse(base_url).netloc:
            return False
        return True