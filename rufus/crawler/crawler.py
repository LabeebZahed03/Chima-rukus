from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
from queue import PriorityQueue
import json
import re
import hashlib
from rufus.prompt_processor.prompt_processor import PromptProcessor
from rufus.model.hcm_model import HybridConditionalModel

class Crawler:
    def __init__(self, instructions: str, max_depth: int = 3, max_retries: int = 3, timeout: int = 10):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.instructions = instructions
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Technical content markers
        self.tech_patterns = {
            'react': [r'react', r'jsx', r'component', r'hook', r'props', r'state'],
            'web_dev': [r'html', r'css', r'javascript', r'frontend', r'backend', r'api'],
            'sql': [r'database', r'query', r'table', r'select', r'insert', r'update']
        }
        
        # Initialize NLP components
        try:
            self.prompt_processor = PromptProcessor(instructions)
            self.query = self.prompt_processor.process_prompt()
            self.keywords = self.query['keywords']
            self.entities = self.query.get('entities', {})
            
            # Add technical terms to keywords
            for category, patterns in self.tech_patterns.items():
                if any(term in instructions.lower() for term in [category, *patterns]):
                    self.keywords.extend(patterns)
            
            self.keywords = list(set(self.keywords))  # Remove duplicates
            self.model = HybridConditionalModel()
            
            self.logger.info(f"Extracted keywords: {self.keywords}")
            self.logger.info(f"Extracted entities: {self.entities}")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")
            raise

        # RAG-specific data structures
        self.document_store = []
        self.metadata_store = {}
        self.content_chunks = defaultdict(list)
        self.relevance_scores = {}
        
        # Crawling state
        self.visited_urls = set()
        self.urls_to_visit = PriorityQueue()
        
        # Set up Selenium
        self._setup_selenium()
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Fetching {url} (attempt {attempt + 1})")
                self.driver.get(url)
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(1)  # Short wait for dynamic content
                
                # Get the page source
                content = self.driver.page_source
                
                # Store raw HTML with the chunks
                if content:
                    chunk = {
                        'id': hashlib.md5(url.encode()).hexdigest(),
                        'html': content,  # Store raw HTML
                        'url': url,
                        'type': 'html',
                        'timestamp': time.time()
                    }
                    self.content_chunks[url].append(chunk)
                    
                return content
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1)  # Wait before retry
                
        return None

    def _setup_selenium(self):
        """Set up Selenium with modern browser configuration."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
            self.logger.info("Selenium setup completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Selenium: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove common boilerplate text
            boilerplate = [
                'cookie policy',
                'privacy policy',
                'terms of service',
                'accept cookies',
                'advertisement',
            ]
            for phrase in boilerplate:
                text = text.replace(phrase, '')
            
            # Clean special characters but preserve code syntax
            text = re.sub(r'[^\w\s\-_.,!?(){}[\]<>=+*/\'\":]', ' ', text)
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return text
    def _get_initial_relevant_links(self) -> List[Tuple[float, str]]:
        """Get highly relevant links from the start page."""
        relevant_links = []
        
        # Look for technical article links
        article_patterns = [
            r'/blog/',
            r'/articles/',
            r'/tutorials/',
            r'/react',
            r'/javascript',
            r'/sql',
            r'/web-development'
        ]
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True).lower()
            
            # Skip navigation links
            if self._is_navigation_element(link.parent):
                continue
                
            # Calculate relevance
            relevance = 0
            
            # Check URL patterns
            if any(pattern in href.lower() for pattern in article_patterns):
                relevance += 0.3
                
            # Check for technical keywords
            if any(keyword in text for keyword in self.keywords):
                relevance += 0.4
                
            # Check context
            parent_text = link.parent.get_text(strip=True).lower()
            if any(keyword in parent_text for keyword in self.keywords):
                relevance += 0.2
                
            if relevance > 0.4:  # Only keep highly relevant links
                relevant_links.append((relevance, href))
                
        # Sort by relevance
        relevant_links.sort(reverse=True)
        return relevant_links[:10]
    
    def _queue_relevant_links(self, soup: BeautifulSoup, current_url: str, depth: int):
        """Queue relevant links for crawling."""
        try:
            # Get relevant URLs
            relevant_urls = self._extract_relevant_urls(soup, current_url)
            
            # Add to priority queue
            for url, score in relevant_urls:
                if url not in self.visited_urls:
                    # Add as negative score for max-heap behavior
                    self.urls_to_visit.put((-score, score, url, depth + 1))
                    self.logger.debug(f"Queued {url} with score {score}")
                    
        except Exception as e:
            self.logger.error(f"Error queuing links from {current_url}: {e}")
    
    def _queue_url(self, url: str, score: float, depth: int):
        """
        Add URL to priority queue with standardized format.
        
        :param url: URL to queue
        :param score: Relevance score (0-1)
        :param depth: Current depth
        """
        try:
            if url not in self.visited_urls:
                # Store as (priority, score, url, depth)
                # Priority is negative score for max-heap behavior
                self.urls_to_visit.put((-score, score, url, depth))
                self.logger.debug(f"Queued {url} with score {score} at depth {depth}")
        except Exception as e:
            self.logger.error(f"Error queuing URL {url}: {e}")

    def crawl(self, start_url: str) -> List[Dict[str, Any]]:
        """
        Crawl website with better controls and efficiency.
        """
        self.logger.info(f"Starting RAG-focused crawl from: {start_url}")
        
        # Set limits
        MAX_PAGES = 20  # Maximum pages to process
        MAX_TIME = 300  # Maximum time in seconds (5 minutes)
        start_time = time.time()
        
        try:
            # Process start URL
            self._process_url(start_url, depth=0)
            
            # Initialize priority queue with initial relevant links
            initial_links = self._get_initial_relevant_links()
            if len(initial_links) == 0:
                self.logger.warning("No relevant links found on start page")
            else:
                # Queue initial links
                for score, url in initial_links:
                    self._queue_url(url, score, depth=1)
                self.logger.info(f"Queued {len(initial_links)} initial links")
            
            processed_count = 1  # Count start URL
            
            # Process queue with limits
            while (not self.urls_to_visit.empty() and 
                processed_count < MAX_PAGES and 
                time.time() - start_time < MAX_TIME):
                
                # Get next URL from queue (priority, score, url, depth)
                _, score, url, depth = self.urls_to_visit.get()
                
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                    
                # Only process highly relevant pages
                if score > 0.4:  # Increased relevance threshold
                    self._process_url(url, depth)
                    processed_count += 1
                    self.logger.info(f"Processed {processed_count} pages")

            # Generate final documents
            documents = self._generate_rag_documents()
            
            self.logger.info(f"Crawl completed. Found {len(documents)} relevant documents")
            if len(documents) == 0:
                self.logger.warning("No relevant content found. Consider adjusting thresholds")
                
            return documents
            
        except Exception as e:
            self.logger.error(f"Error during crawl: {e}")
            return []
        finally:
            self.driver.quit()

    def _extract_relevant_urls(self, soup: BeautifulSoup, current_url: str) -> List[Tuple[str, float]]:
        """Extract and score relevant URLs from the page."""
        relevant_urls = []
        base_domain = urlparse(current_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            
            # Skip if different domain or already visited
            if (urlparse(absolute_url).netloc != base_domain or 
                absolute_url in self.visited_urls):
                continue
                
            # Skip media and document files
            if re.search(r'\.(jpg|jpeg|png|gif|pdf|doc|docx)$', absolute_url, re.I):
                continue
                
            # Get link text and context
            link_text = link.get_text(strip=True)
            context_text = link.parent.get_text(strip=True) if link.parent else ""
            
            # Calculate relevance
            url_score = self._calculate_url_relevance(absolute_url, link_text, context_text)
            
            if url_score > 0.2:
                relevant_urls.append((absolute_url, url_score))
                
        return relevant_urls

    
    def _process_article_content(self, url: str, article: BeautifulSoup):
        """Process article content more efficiently."""
        # Extract text content
        text_content = self._clean_text(article.get_text())
        if text_content:
            self.content_chunks[url].append({
                'id': hashlib.md5(text_content.encode()).hexdigest(),
                'content': text_content,
                'type': 'text',
                'relevance': self._calculate_relevance_score(text_content)
            })
        
        # Extract code blocks
        for code_block in article.find_all(['pre', 'code']):
            code = self._clean_text(code_block.get_text())
            if code:
                self.content_chunks[url].append({
                    'id': hashlib.md5(code.encode()).hexdigest(),
                    'content': code,
                    'type': 'code',
                    'language': self._detect_code_language(code_block),
                    'relevance': self._calculate_relevance_score(code) + 0.2
                })

    def _process_url(self, url: str, depth: int):
        """Process a single URL with better content extraction."""
        if url in self.visited_urls:
            return
            
        self.logger.info(f"Processing URL: {url} at depth {depth}")
        self.visited_urls.add(url)
        
        try:
            content = self._fetch_page(url)
            if not content:
                return
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Store page source
            self.content_chunks[url].append({
                'id': hashlib.md5(url.encode()).hexdigest(),
                'html': content,
                'url': url,
                'type': 'html',
                'timestamp': time.time()
            })
            
            # Extract relevant content
            article_content = soup.find('article') or soup.find(class_=lambda x: x and 'article' in x.lower())
            if article_content:
                self._process_article_content(url, article_content)
                
            # Find next relevant links
            if depth < self.max_depth:
                self._queue_relevant_links(soup, url, depth)
                
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
    
    def _detect_code_language(self, element: BeautifulSoup) -> str:
        """Detect programming language of code block."""
        # Check class attributes for language hints
        classes = element.get('class', [])
        for cls in classes:
            if cls.startswith(('language-', 'lang-')):
                return cls.split('-')[1]
                
        # Check content for language patterns
        text = element.get_text()
        patterns = {
            'python': r'def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import',
            'javascript': r'function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=',
            'html': r'<\/?[a-z][\s\S]*>',
            'css': r'{\s*[\w-]+\s*:',
            'sql': r'SELECT\s+|INSERT\s+INTO|CREATE\s+TABLE',
            'react': r'function\s+\w+\s*\(\)\s*{\s*return\s*<|class\s+\w+\s+extends\s+React'
        }
        
        for lang, pattern in patterns.items():
            if re.search(pattern, text, re.I):
                return lang
                
        return 'unknown'
    
    def _is_navigation_element(self, element: BeautifulSoup) -> bool:
        """Check if element is likely navigation."""
        nav_indicators = ['nav', 'menu', 'header', 'footer', 'sidebar', 'navigation']
        
        # Check element classes and ID
        element_classes = ' '.join(element.get('class', []))
        element_id = element.get('id', '').lower()
        
        # Check role attribute
        role = element.get('role', '').lower()
        
        return any(ind in (element_classes + ' ' + element_id + ' ' + role).lower() 
                for ind in nav_indicators)

    def _calculate_url_relevance(self, url: str, link_text: str = "", context_text: str = "") -> float:
        """Calculate URL relevance score."""
        score = 0.0
        
        # Check URL path
        path = urlparse(url).path.lower()
        path_segments = [seg for seg in path.split('/') if seg]
        
        for keyword in self.keywords:
            if keyword in path.lower():
                score += 0.3
                
        # Check link text and context
        for keyword in self.keywords:
            if keyword in link_text.lower():
                score += 0.4
            if keyword in context_text.lower():
                score += 0.2
                
        # Boost for technical paths
        tech_indicators = ['blog', 'article', 'tutorial', 'guide', 'doc', 'react', 'javascript', 'sql']
        if any(ind in path.lower() for ind in tech_indicators):
            score += 0.2
            
        return min(1.0, score)
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate content relevance score."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(2.0 for keyword in self.keywords 
                            if keyword in text_lower)
        score += keyword_matches / (len(self.keywords) + 1)
        
        # Look for code patterns
        if re.search(r'\b\w+\s*\(\s*\)', text):  # Function-like patterns
            score += 0.2
        if re.search(r'<[^>]+>', text):  # HTML/XML-like patterns
            score += 0.2
        if re.search(r'{\s*[\w-]+\s*:', text):  # CSS-like patterns
            score += 0.2
        if re.search(r'(SELECT|INSERT|UPDATE|CREATE)\s+\w+', text, re.I):  # SQL-like patterns
            score += 0.2
            
        # Content quality indicators
        words = text.split()
        if len(words) > 50:  # Substantial content
            score += 0.1
        if len(set(words)) / len(words) > 0.4:  # Vocabulary diversity
            score += 0.1
            
        return min(1.0, score)

    def _extract_content_chunks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract content in chunks suitable for RAG systems."""
        chunks = []
        
        # Process article-like content
        for element in soup.find_all(['article', 'section', 'div', 'main']):
            # Skip navigation and irrelevant elements
            if self._is_navigation_element(element):
                continue
                
            text = self._clean_text(element.get_text(strip=True))
            if not text or len(text) < 100:  # Skip short content
                continue
                
            relevance = self._calculate_relevance_score(text)
            if relevance > 0.2:  # Lower threshold for initial extraction
                chunk = {
                    'id': hashlib.md5(text.encode()).hexdigest(),
                    'content': text,
                    'type': 'text',
                    'relevance': relevance,
                    'word_count': len(text.split()),
                    'html': str(element),  # Keep original HTML
                    'tag': element.name,
                    'classes': element.get('class', []),
                }
                chunks.append(chunk)
        
        # Process code blocks
        for code_block in soup.find_all(['pre', 'code']):
            text = self._clean_text(code_block.get_text(strip=True))
            if text:
                chunk = {
                    'id': hashlib.md5(text.encode()).hexdigest(),
                    'content': text,
                    'type': 'code',
                    'relevance': self._calculate_relevance_score(text) + 0.2,  # Boost code relevance
                    'language': self._detect_code_language(code_block),
                    'html': str(code_block)
                }
                chunks.append(chunk)
        
        # Process tables
        for table in soup.find_all('table'):
            chunk = self._process_table(table)
            if chunk:
                chunks.append(chunk)
                
        return chunks


    def _is_content_element(self, element: BeautifulSoup) -> bool:
        """Determine if an element contains relevant content."""
        if not element:
            return False
            
        # Skip navigation elements
        if self._is_navigation_element(element):
            return False
            
        text = element.get_text(strip=True)
        if not text or len(text) < 100:  # Minimum content length
            return False
            
        # Check relevance
        relevance = self._calculate_relevance_score(text)
        return relevance > 0.2

    def _process_content_element(self, element: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Process a content element into a RAG-suitable chunk."""
        text = self._clean_text(element.get_text(strip=True))
        if not text:
            return None
            
        # Generate chunk ID
        chunk_id = hashlib.md5(text.encode()).hexdigest()
        
        # Calculate relevance
        relevance = self._calculate_relevance_score(text)
        
        # Boost relevance for technical content
        if any(indicator in str(element).lower() for indicator in 
            ['code', 'pre', 'script', 'sql', 'react', 'javascript']):
            relevance += 0.2
            
        # Skip if still not relevant enough
        if relevance < 0.2:  # Lowered threshold
            return None
            
        return {
            'id': chunk_id,
            'content': text,
            'relevance': relevance,
            'type': 'text',
            'word_count': len(text.split()),
            'html': str(element),  # Keep original HTML
            'tag': element.name,
            'classes': element.get('class', []),
            'metadata': {
                'has_code': bool(element.find(['code', 'pre'])),
                'technical_indicators': [
                    ind for ind in ['react', 'javascript', 'sql', 'code'] 
                    if ind in str(element).lower()
                ]
            }
        }

    def _process_table(self, table: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Process a table into a structured format."""
        headers = []
        rows = []
        
        # Extract headers
        for th in table.find_all('th'):
            headers.append(th.get_text(strip=True))
        
        # Extract rows
        for tr in table.find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row and not all(cell == '' for cell in row):  # Skip empty rows
                rows.append(row)
                
        if not rows:
            return None
            
        # Calculate relevance
        content_text = ' '.join([' '.join(row) for row in rows])
        relevance = self._calculate_relevance_score(content_text)
        
        if relevance < 0.2:  # Skip irrelevant tables
            return None
            
        return {
            'id': hashlib.md5(str(rows).encode()).hexdigest(),
            'type': 'table',
            'headers': headers,
            'rows': rows,
            'relevance': relevance,
            'row_count': len(rows),
            'column_count': len(headers) if headers else len(rows[0]) if rows else 0,
            'html': str(table)  # Keep original HTML
        }

    def _generate_rag_documents(self) -> List[Dict[str, Any]]:
        """Generate final documents formatted for RAG systems with better error handling."""
        documents = []
        
        try:
            for url, chunks in self.content_chunks.items():
                try:
                    # Get the HTML content first
                    html_chunk = next((chunk for chunk in chunks if chunk['type'] == 'html'), None)
                    
                    if not html_chunk:
                        self.logger.warning(f"No HTML content found for {url}")
                        continue
                    
                    # Filter relevant content chunks
                    content_chunks = [c for c in chunks if c['type'] != 'html' and c.get('relevance', 0) > 0.3]
                    
                    if not content_chunks:
                        self.logger.debug(f"No relevant content chunks found for {url}")
                        continue
                    
                    # Get metadata
                    metadata = self.metadata_store.get(url, {})
                    
                    # Calculate average relevance safely
                    relevance_scores = [c.get('relevance', 0) for c in content_chunks]
                    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
                    
                    # Create document with required fields
                    doc = {
                        'url': url,
                        'title': metadata.get('title', ''),
                        'timestamp': metadata.get('timestamp', time.time()),
                        'html': html_chunk.get('html', ''),  # Original HTML
                        'chunks': content_chunks,
                        'metadata': {
                            'chunk_count': len(content_chunks),
                            'average_relevance': avg_relevance,
                            'types': list(set(c.get('type', 'unknown') for c in content_chunks)),
                            'depth': metadata.get('depth', 0)
                        }
                    }
                    
                    # Add technical content if present
                    technical_chunks = [
                        c for c in content_chunks 
                        if c.get('type') in ['code', 'table'] 
                        or c.get('metadata', {}).get('has_code', False)
                    ]
                    
                    if technical_chunks:
                        doc['technical_content'] = technical_chunks
                    
                    documents.append(doc)
                    self.logger.info(f"Generated document for {url} with {len(content_chunks)} chunks")
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunks for {url}: {e}")
                    continue
            
            # Sort documents by relevance
            documents.sort(key=lambda x: x['metadata']['average_relevance'], reverse=True)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error generating RAG documents: {e}")
            return []

    

    def _extract_semantic_info(self, text: str) -> Dict[str, Any]:
        """Extract semantic information using the HCM model."""
        try:
            predictions = self.model.predict(text[:512])  # Limit length for prediction
            
            # Extract entities and their types
            entities = defaultdict(list)
            current_entity = []
            current_type = None
            
            for token, pred in zip(text.split(), predictions):
                if pred.startswith('B-'):
                    if current_entity:
                        entities[current_type].append(' '.join(current_entity))
                    current_entity = [token]
                    current_type = pred[2:]
                elif pred.startswith('I-') and current_type:
                    current_entity.append(token)
                else:
                    if current_entity:
                        entities[current_type].append(' '.join(current_entity))
                    current_entity = []
                    current_type = None
            
            return {
                'entities': dict(entities),
                'token_count': len(predictions),
                'entity_types': list(entities.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting semantic info: {e}")
            return {}