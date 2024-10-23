import spacy
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import hashlib
import re
from ..utils.logger import get_logger

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.logger = get_logger(__name__)

    def generate_crawl_tasks(self, instructions: str):
        doc = self.nlp(instructions)
        keywords = set()

        # Extract keywords from prepositional objects (pobj) and their conjuncts
        for token in doc:
            if token.dep_ == 'pobj':
                # Get the phrase rooted at the object of the preposition
                phrase_tokens = [t.text for t in token.subtree if not t.is_stop and not t.is_punct]
                if phrase_tokens:
                    phrase = ' '.join(phrase_tokens)
                    keywords.add(phrase.lower().strip())
                # Include conjuncts
                for conj in token.conjuncts:
                    conj_tokens = [t.text for t in conj.subtree if not t.is_stop and not t.is_punct]
                    if conj_tokens:
                        conj_phrase = ' '.join(conj_tokens)
                        keywords.add(conj_phrase.lower().strip())

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            keywords.add(chunk.text.lower().strip())

        # Extract individual nouns and proper nouns
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop and not token.is_punct:
                keywords.add(token.lemma_.lower().strip())

        self.logger.info(f"Generated Keywords: {keywords}")
        return keywords

    def parse(self, html: str, instructions: str, url: str, is_xml=False):
        if not html:
            self.logger.warning(f"No HTML content for {url}")
            return {}

        self.logger.info(f"Parsing: {url}")

        soup = BeautifulSoup(html, 'xml' if is_xml else 'html.parser')

        # Generate a unique id for the article based on the URL
        article_id = hashlib.md5(url.encode('utf-8')).hexdigest()

        # Extract source from the URL
        parsed_url = urlparse(url)
        source = parsed_url.netloc

        # Set default language
        language = 'en'

        title = soup.title.string.strip() if soup.title else ''

        author = ''
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and 'content' in author_meta.attrs:
            author = author_meta['content'].strip()

        publication_date = ''
        date_meta = soup.find('meta', attrs={'property': 'article:published_time'})
        if date_meta and 'content' in date_meta.attrs:
            publication_date = date_meta['content'].strip()
        else:
            date_meta = soup.find('meta', attrs={'name': 'date'})
            if date_meta and 'content' in date_meta.attrs:
                publication_date = date_meta['content'].strip()

        if not publication_date:
            publication_date = datetime.now().strftime('%Y-%m-%d')

        tags = []
        tag_elements = soup.find_all('meta', attrs={'property': 'article:tag'})
        for tag_element in tag_elements:
            if 'content' in tag_element.attrs:
                tags.append(tag_element['content'].strip())

        keywords = self.generate_crawl_tasks(instructions)
        print(f"Generated Keywords: {keywords}")

        content_sections = []
        chunk_counter = 1  # Initialize chunk counter

        for header in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = header.get_text(strip=True)
            content = ''
            next_node = header.find_next_sibling()
            while next_node:
                if next_node.name and next_node.name.startswith('h') and int(next_node.name[1]) <= int(header.name[1]):
                    break
                if next_node.name in ['p', 'div', 'ul', 'ol', 'pre', 'code']:
                    content += next_node.get_text(separator=' ', strip=True) + ' '
                next_node = next_node.find_next_sibling()

            if content:
                section_doc = self.nlp(content)
                section_tokens = set(token.lemma_.lower() for token in section_doc if not token.is_stop and not token.is_punct)
                if keywords.intersection(section_tokens):
                    content_sections.append({
                        'chunk_id': f"{article_id}-chunk-{chunk_counter}",
                        'heading': heading_text,
                        'text': content.strip()
                    })
                    chunk_counter += 1

        # If no content matched, consider including all sections
        if not content_sections:
            self.logger.info("No content matched keywords, including all sections.")
            chunk_counter = 1  
            for header in soup.find_all(['h1', 'h2', 'h3']):
                heading_text = header.get_text(strip=True)
                content = ''
                next_node = header.find_next_sibling()
                while next_node:
                    if next_node.name and next_node.name.startswith('h') and int(next_node.name[1]) <= int(header.name[1]):
                        break
                    if next_node.name in ['p', 'div', 'ul', 'ol', 'pre', 'code']:
                        content += next_node.get_text(separator=' ', strip=True) + ' '
                    next_node = next_node.find_next_sibling()
                if content:
                    content_sections.append({
                        'chunk_id': f"{article_id}-chunk-{chunk_counter}",
                        'heading': heading_text,
                        'text': content.strip()
                    })
                    chunk_counter += 1

        extracted_data = {
            'id': article_id,
            'url': url,
            'title': title,
            'author': author if author else '',
            'publication_date': publication_date,
            'tags': tags,
            'source': source,
            'language': language,
            'chunks': content_sections
        }

        return extracted_data
