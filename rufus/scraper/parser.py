# Rufus/scraper/parser.py

import spacy
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        # Optionally, adjust max_length if necessary (not recommended)
        # self.nlp.max_length = 1500000  # Increase limit if needed

    def generate_crawl_tasks(self, instructions: str):
        input_text = f"Extract keywords: {instructions}"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        keywords = [kw.strip().lower() for kw in generated_text.split(',')]
        return set(keywords)

    def parse(self, html: str, instructions: str, is_xml=False):
        if not html:
            return {}
        if is_xml:
            soup = BeautifulSoup(html, 'xml')  # Use the XML parser
        else:
            soup = BeautifulSoup(html, 'html.parser')
        soup = BeautifulSoup(html, 'html.parser')
        extracted_data = {}

        # Generate keywords using the pre-trained model
        keywords = self.generate_crawl_tasks(instructions)
        print(f"Generated Keywords: {keywords}")

        # Extract data based on keywords
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Split the text into smaller chunks (e.g., paragraphs)
        paragraphs = text_content.split('\n')
        
        relevant_content = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue  # Skip empty paragraphs
            if len(paragraph) > 100000:
                # If paragraph is still too long, skip or further split it
                continue
            try:
                doc = self.nlp(paragraph)
                for sent in doc.sents:
                    sentence = sent.text.strip()
                    if any(kw in sentence.lower() for kw in keywords):
                        relevant_content.append(sentence)
            except Exception as e:
                print(f"Error processing paragraph: {e}")
                continue

        extracted_data['relevant_content'] = relevant_content
        return extracted_data
