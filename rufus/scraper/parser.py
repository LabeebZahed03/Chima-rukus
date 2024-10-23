import spacy
from bs4 import BeautifulSoup

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def parse_instructions(self, instructions: str):
        doc = self.nlp(instructions)
        keywords = [token.lemma_.lower() for token in doc if token.pos_ in ('NOUN', 'PROPN')]
        return set(keywords)

    def compute_relevance(self, text: str, keywords: set):
        doc = self.nlp(text)
        relevance = sum(1 for token in doc if token.lemma_.lower() in keywords)
        return relevance / len(doc) if len(doc) > 0 else 0

    def parse(self, html: str, instructions: str):
        if not html:
            return {}

        soup = BeautifulSoup(html, 'html.parser')
        extracted_data = {}

        # Parse instructions to get keywords
        keywords = self.parse_instructions(instructions)

        # Extract data based on keywords
        text_content = soup.get_text(separator=' ', strip=True)
        paragraphs = text_content.split('\n')

        relevant_content = []
        for para in paragraphs:
            relevance = self.compute_relevance(para, keywords)
            if relevance > 0.1:  # Threshold can be adjusted
                relevant_content.append(para)

        extracted_data['relevant_content'] = relevant_content
        return extracted_data