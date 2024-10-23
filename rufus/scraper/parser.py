
import spacy
from bs4 import BeautifulSoup

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

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

    def parse(self, html: str, instructions: str, is_xml=False):
        if not html:
            return {}

        soup = BeautifulSoup(html, 'xml' if is_xml else 'html.parser')
        extracted_data = {}

        # Generate keywords using spaCy
        keywords = self.generate_crawl_tasks(instructions)
        print(f"Generated Keywords: {keywords}")

        # Extract data based on keywords
        text_content = soup.get_text(separator=' ', strip=True)

        # Split text into manageable chunks
        paragraphs = text_content.split('\n')

        relevant_content = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue  # Skip empty paragraphs
            if len(paragraph) > 100000:
                # Skip overly long paragraphs
                continue
            paragraph_doc = self.nlp(paragraph)
            paragraph_tokens = set([token.lemma_.lower() for token in paragraph_doc])

            if keywords.intersection(paragraph_tokens):
                relevant_content.append(paragraph)

        extracted_data['relevant_content'] = relevant_content
        return extracted_data
