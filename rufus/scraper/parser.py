from bs4 import BeautifulSoup

class Parser:
    def parse(self, html: str, instructions: str):
        if not html:
            return {}

        soup = BeautifulSoup(html, 'html.parser')

        # Placeholder parsing logic based on instructions
        if "title" in instructions.lower():
            title = soup.title.string if soup.title else 'No title found'
            return {'title': title}
        else:
            # Extract all text
            text = soup.get_text(separator=' ', strip=True)
            return {'text': text}