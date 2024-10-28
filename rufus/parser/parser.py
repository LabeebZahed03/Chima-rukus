from bs4 import BeautifulSoup
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from rufus.model.hcm_model import HybridConditionalModel
from rufus.prompt_processor.prompt_processor import PromptProcessor
import time
import hashlib

logging.basicConfig(level=logging.INFO)

class Parser:
    def __init__(self):
        """Initialize the Parser with required NLP models and logging."""
        self.model = HybridConditionalModel()
        self.prompt_processor = PromptProcessor("")
        self.logger = logging.getLogger(__name__)
        
    def parse(self, html: str, instructions: str, url: str, is_xml: bool = False) -> Dict[str, Any]:
        """Parse the HTML content and structure for RAG usage."""
        try:
            self.prompt_processor.instructions = instructions
            query = self.prompt_processor.process_prompt()
            keywords = query['keywords']
            entities = query['entities']

            soup = BeautifulSoup(html, 'html.parser')
            self.logger.info(f"Parsing URL: {url}")

            # Find main content area
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find(class_=lambda x: x and 'article' in str(x).lower()) or
                soup.find(class_=lambda x: x and 'content' in str(x).lower())
            )

            if not main_content:
                main_content = soup.body if soup.body else soup

            # Extract metadata
            metadata = self._extract_metadata(soup)
            metadata.update({
                'url': url,
                'title': self._extract_title(soup),
                'source_type': 'web',
                'crawl_time': time.time(),
                'keywords': keywords
            })

            # Get relevant sections
            tree = self._build_html_tree(main_content)
            potential_sections = self._quick_section_scan(tree, keywords)
            
            # Process most relevant sections
            relevant_sections = []
            for section in potential_sections[:50]:  # Limit to top 50 sections
                score = self._calculate_semantic_score(section['text'], keywords, entities)
                if score > 0.3:
                    section['score'] = score
                    relevant_sections.append(section)

            # Process into chunks
            chunks = self._process_relevant_sections(relevant_sections, keywords)
            
            # Calculate overall relevance
            relevance_score = 0.0
            if chunks:
                chunk_scores = [c.get('score', 0) for c in chunks]
                relevance_score = sum(chunk_scores) / len(chunk_scores)

            # Structure final output
            output = {
                'metadata': metadata,
                'relevance_score': relevance_score,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'content_types': list(set(c.get('chunk_type', 'text') for c in chunks)),
                'technical_concepts': list(set(
                    concept 
                    for chunk in chunks 
                    for concept in chunk.get('metadata', {}).get('technical_indicators', [])
                ))
            }

            return output

        except Exception as e:
            self.logger.error(f"Error parsing {url}: {e}")
            return {
                'metadata': {'url': url, 'error': str(e)},
                'chunks': [],
                'relevance_score': 0.0
            }

    def _quick_section_scan(self, tree: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
        """Quickly scan sections for potential relevance without deep analysis."""
        potential_sections = []
        
        def scan_tree(node: Dict[str, Any], depth: int = 0):
            text = node.get('text', '')
            if text and len(text.split()) >= 10:  # Minimum length check
                score = self._quick_relevance_check(text, keywords)
                if score > 0.2:  # Lower threshold for initial scan
                    potential_sections.append({
                        'text': text,
                        'tag': node['tag'],
                        'attributes': node['attributes'],
                        'quick_score': score
                    })
            
            for child in node.get('children', []):
                scan_tree(child, depth + 1)
        
        scan_tree(tree)
        return sorted(potential_sections, key=lambda x: x['quick_score'], reverse=True)

    def _quick_relevance_check(self, text: str, keywords: List[str]) -> float:
        """Quick relevance check without using transformer models."""
        if not text:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        # Keyword matching
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 0.3
        
        # Technical indicators
        if any(pattern in text_lower for pattern in [
            'function', 'class', 'import', 'const',
            'react', 'component', 'hook', 'jsx',
            'sql', 'query', 'table', 'select'
        ]):
            score += 0.2
            
        # Content quality
        words = text.split()
        if len(words) > 50:  # Substantial content
            score += 0.1
        if len(set(words)) / (len(words) + 1) > 0.4:  # Vocabulary diversity
            score += 0.1
            
        return min(1.0, score)

    def _build_html_tree(self, element: BeautifulSoup) -> Dict[str, Any]:
        """Build a tree representation of HTML content."""
        tree = {
            'tag': element.name,
            'text': element.get_text(strip=True),
            'attributes': element.attrs if hasattr(element, 'attrs') else {},
            'children': []
        }
        
        if hasattr(element, 'children'):
            for child in element.children:
                if isinstance(child, str):
                    continue
                child_tree = self._build_html_tree(child)
                if child_tree['text']:  # Only add non-empty nodes
                    tree['children'].append(child_tree)
        
        return tree

    def _calculate_semantic_score(
        self, 
        text: str, 
        keywords: List[str],
        entities: Dict[str, List[str]]
    ) -> float:
        """Calculate semantic relevance score using HCM model and keyword matching."""
        if not text:
            return 0.0
            
        # Get predictions from HCM model for a limited sample
        sample_text = ' '.join(text.split()[:200])  # Limit text length
        predictions = self.model.predict(sample_text)
        
        # Calculate base keyword score
        keyword_score = sum(2.0 for keyword in keywords 
                          if keyword.lower() in text.lower()) / (len(keywords) + 1)
        
        # Calculate entity score
        entity_score = 0.0
        for entity_type, entity_values in entities.items():
            for entity in entity_values:
                if entity.lower() in text.lower():
                    entity_score += 1.0
        entity_score = entity_score / (sum(len(v) for v in entities.values()) + 1)
        
        # Combine scores with weights
        return 0.5 * keyword_score + 0.3 * entity_score + 0.2 * (len(predictions) / 100)

    # [Previous helper methods remain the same]
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title with fallbacks."""
        title = soup.title.string if soup.title else None
        if not title:
            h1 = soup.find('h1')
            title = h1.get_text(strip=True) if h1 else None
        return title

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract relevant metadata from the page."""
        metadata = {}
        meta_tags = soup.find_all('meta')
        
        for tag in meta_tags:
            name = tag.get('name', '').lower()
            content = tag.get('content', '')
            if name in ['description', 'keywords', 'author', 'date', 'published']:
                metadata[name] = content
                
        return metadata
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving code blocks."""
        sentences = []
        # Split by common sentence endings but preserve code patterns
        patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Regular sentences
            r'(?<=[;])\s+(?=\w)',       # Code statements
            r'(?<=\})\s*(?=\w)',        # Code blocks
            r'(?<=\n)\s*(?=[A-Z])'      # New lines starting with capital
        ]
        
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            next_split = text_length
            matching_pattern = None
            
            # Find next sentence boundary
            for pattern in patterns:
                match = re.search(pattern, text[current_pos:])
                if match and current_pos + match.start() < next_split:
                    next_split = current_pos + match.start()
                    matching_pattern = pattern
            
            # Extract sentence
            sentence = text[current_pos:next_split].strip()
            if sentence:
                sentences.append(sentence)
            
            # Move position
            current_pos = next_split + 1
            if matching_pattern and current_pos < text_length:
                # Skip whitespace
                while current_pos < text_length and text[current_pos].isspace():
                    current_pos += 1
        
        return [s for s in sentences if s]
    
    def _split_long_sentence(self, sentence: str, max_length: int) -> List[str]:
        """Split long sentences while preserving meaning."""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word.split())
            if current_length + word_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    

    
    def _create_semantic_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into semantic chunks suitable for embeddings."""
        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If single sentence is too long, split it
            if sentence_length > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Split long sentence into smaller parts
                sentence_chunks = self._split_long_sentence(sentence, max_chunk_size)
                chunks.extend(sentence_chunks)
                continue
                
            # Check if adding sentence exceeds max size
            if current_size + sentence_length > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _classify_chunk_type(self, text: str) -> str:
        """Classify chunk type based on content."""
        text_lower = text.lower()
        
        # Code block detection
        code_patterns = [
            (r'function\s+\w+\s*\(.*\)', 'code_function'),
            (r'class\s+\w+', 'code_class'),
            (r'import\s+[\w\s,{}]+\s+from', 'code_import'),
            (r'const\s+\w+\s*=', 'code_variable'),
            (r'<[^>]+>', 'code_markup'),
            (r'SELECT\s+.*\s+FROM', 'code_sql')
        ]
        
        for pattern, chunk_type in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return chunk_type
        
        # Technical content detection
        if any(indicator in text_lower for indicator in [
            'api', 'function', 'method', 'parameter', 'return',
            'component', 'props', 'state', 'hook', 'effect',
            'query', 'database', 'table', 'column'
        ]):
            return 'technical_content'
        
        # Example/tutorial detection
        if any(indicator in text_lower for indicator in [
            'example:', 'for example', 'sample', 'tutorial',
            'step 1', 'first,', 'here\'s how'
        ]):
            return 'example'
            
        return 'text'

    def _detect_technical_indicators(self, text: str) -> List[str]:
        """Detect technical concepts and patterns in text."""
        indicators = []
        text_lower = text.lower()
        
        patterns = {
            'react': ['component', 'props', 'state', 'hook', 'jsx'],
            'javascript': ['function', 'const', 'let', 'var', 'async'],
            'sql': ['select', 'from', 'where', 'join', 'table'],
            'api': ['endpoint', 'request', 'response', 'http', 'api'],
            'data': ['array', 'object', 'string', 'number', 'boolean']
        }
        
        for category, terms in patterns.items():
            if any(term in text_lower for term in terms):
                indicators.append(category)
        
        return indicators

    def _process_relevant_sections(
        self, 
        sections: List[Dict[str, Any]], 
        keywords: List[str]) -> List[Dict[str, Any]]:
        """Process and structure relevant sections into RAG-friendly chunks."""
        processed_chunks = []
        
        for section in sections:
            # Split content into smaller chunks
            chunks = self._create_semantic_chunks(section['text'])
            
            for chunk_text in chunks:
                # Create chunk metadata
                chunk = {
                    'content': chunk_text,
                    'chunk_type': 'text',
                    'tag': section['tag'],
                    'score': section['score'],
                    'word_count': len(chunk_text.split()),
                    'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest(),
                    'metadata': {
                        'source_tag': section['tag'],
                        'keywords_found': [k for k in keywords if k.lower() in chunk_text.lower()],
                        'attributes': section['attributes'],
                        'technical_indicators': self._detect_technical_indicators(chunk_text)
                    }
                }
                
                # Add chunk type classification
                chunk_type = self._classify_chunk_type(chunk_text)
                if chunk_type:
                    chunk['chunk_type'] = chunk_type
                
                processed_chunks.append(chunk)
                
        return processed_chunks

    def _calculate_page_relevance(
        self, 
        content_sections: List[Dict[str, Any]], 
        total_keywords: int
    ) -> float:
        """Calculate overall page relevance score."""
        if not content_sections:
            return 0.0
            
        section_scores = [section['score'] for section in content_sections]
        weights = np.array([1.0 / (i + 1) for i in range(len(section_scores))])
        weighted_score = np.sum(weights * section_scores) / np.sum(weights)
        
        return min(1.0, weighted_score)

    def _extract_entities_with_context(
        self,
        content_sections: List[Dict[str, Any]],
        entities_dict: Dict[str, List[Any]],
        keywords: List[str],
        query_entities: Dict[str, List[str]]
    ):
        """Extract entities with surrounding context using HCM model."""
        for section in content_sections:
            # Limit text length for entity extraction
            text = ' '.join(section['content'].split()[:200])
            predictions = self.model.predict(text)
            
            current_entity = []
            current_type = None
            
            for token, pred in zip(text.split(), predictions):
                if pred.startswith('B-'):
                    if current_entity:
                        self._add_entity_to_dict(
                            ' '.join(current_entity),
                            current_type,
                            section['context'],
                            entities_dict
                        )
                    current_entity = [token]
                    current_type = pred[2:]
                elif pred.startswith('I-') and current_type:
                    current_entity.append(token)
                else:
                    if current_entity:
                        self._add_entity_to_dict(
                            ' '.join(current_entity),
                            current_type,
                            section['context'],
                            entities_dict
                        )
                    current_entity = []
                    current_type = None

    def _add_entity_to_dict(
        self,
        entity: str,
        entity_type: str,
        context: Dict[str, Any],
        entities_dict: Dict[str, List[Any]]
    ):
        """Add extracted entity with context to entities dictionary."""
        if not entity or not entity_type:
            return
            
        entities_dict[entity_type].append({
            'value': entity,
            'context': context,
            'confidence': self._calculate_semantic_score(entity, [entity], {})
        })