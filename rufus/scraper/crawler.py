import requests

class Crawler:
    def __init__(self):
        self.session = requests.Session()

    def crawl(self, url: str):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text  # Return the raw HTML
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None