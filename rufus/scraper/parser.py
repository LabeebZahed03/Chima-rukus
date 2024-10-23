import spacy
from bs4 import BeautifulSoup
from datetime import datetime
from ..utils.logger import get_logger

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.logger = get_logger(__name__)

    def generate_crawl_tasks(self, instructions: str):
        doc = self.nlp(instructions)
        keywords = set()
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            keywords.add(chunk.lemma_.lower())
        # Extract named entities
        for ent in doc.ents:
            keywords.add(ent.lemma_.lower())
        return keywords

    def parse(self, html: str, instructions: str, url: str, is_xml=False):
        if not html:
            self.logger.warning(f"No HTML content for {url}")
            return {}

        self.logger.info(f"Parsing: {url}")

        soup = BeautifulSoup(html, 'xml' if is_xml else 'html.parser')

        # Extract metadata
        title = soup.title.string.strip() if soup.title else ''
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        last_updated = ''

        # last updated date in meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name', '').lower() in ['last-modified', 'last_updated', 'date']:
                last_updated = meta.get('content', '')
                break

        # current date if not found
        if not last_updated:
            last_updated = datetime.now().isoformat()

        # Generate keywords using spaCy
        keywords = self.generate_crawl_tasks(instructions)
        print(f"Generated Keywords: {keywords}")

        # Extract data based on keywords
        text_content = soup.get_text(separator=' ', strip=True)
        paragraphs = text_content.split('\n')

        relevant_content = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > 100000:
                continue
            paragraph_doc = self.nlp(paragraph)
            paragraph_tokens = set(token.lemma_.lower() for token in paragraph_doc)

            if keywords.intersection(paragraph_tokens):
                relevant_content.append(paragraph)

        # Structure the extracted data
        extracted_data = {
            'url': url,
            'title': title,
            'headings': headings,
            'last_updated': last_updated,
            'content': relevant_content
        }

        return extracted_data