import spacy
import logging
from typing import Dict, List, Tuple, Any

class PromptProcessor:
    def __init__(self, instructions: str):
        """
        Initialize the PromptProcessor with user instructions.

        :param instructions: User's instructions as a string.
        """
        self.instructions = instructions
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.error(f"Error loading spaCy model: {e}")
            raise

    def extract_keywords_and_entities(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Extract relevant keywords and named entities from the prompt.

        :return: Tuple of (keywords list, entities dictionary)
        """
        doc = self.nlp(self.instructions)
        keywords = set()
        entities = {}

        # Extract keywords based on POS tags
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                keywords.add(token.lemma_.lower())

        # Extract named entities
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            if entity_type not in entities:
                entities[entity_type] = set()
            entities[entity_type].add(ent.text.lower())

        return list(keywords), {k: list(v) for k, v in entities.items()}

    def process_prompt(self) -> Dict[str, Any]:
        """
        Process the prompt to extract keywords and generate a structured query.

        :return: Structured query dictionary
        """
        try:
            keywords, entities = self.extract_keywords_and_entities()
            
            # Create structured query
            query = {
                "keywords": keywords,
                "entities": entities,
                "instructions_summary": self.instructions
            }
            
            return query
        except Exception as e:
            logging.error(f"Error processing prompt: {e}")
            raise