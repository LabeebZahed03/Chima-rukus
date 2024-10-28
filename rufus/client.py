import os
import json
from datetime import datetime
import logging
from typing import Dict, Any, List

from rufus.crawler.crawler import Crawler
from rufus.parser.parser import Parser

class RufusClient:
    def __init__(self, api_key: str):
        """
        Initialize the RufusClient with an API key and set up parser and logging.
        """
        self.api_key = api_key
        if not self._validate_api_key():
            raise ValueError("Invalid API key.")
        
        self.parser = Parser()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _validate_api_key(self) -> bool:
        """
        Placeholder for API key validation logic. Currently accepts any non-empty string.
        """
        return bool(self.api_key)

    def scrape(self, url: str, instructions: str, depth: int = 3, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Crawl a website and parse content based on instructions.
        """
        self.logger.info(f"Starting scrape for URL: {url} with instructions: '{instructions}'")
        
        try:
            # Initialize crawler
            crawler = Crawler(instructions=instructions, max_depth=depth)
            
            # Get crawled pages
            crawled_data = crawler.crawl(url)
            
            # Process each page with the parser
            structured_data = []
            
            if not crawler.content_chunks:
                self.logger.warning("No content chunks found during crawl")
                return []
                
            # Process each URL's content
            for url, chunks in crawler.content_chunks.items():
                try:
                    # Find the HTML content
                    html_chunk = next((chunk for chunk in chunks if chunk['type'] == 'html'), None)
                    
                    if html_chunk and html_chunk.get('html'):
                        # Parse the content
                        parsed_data = self.parser.parse(
                            html=html_chunk['html'],
                            instructions=instructions,
                            url=url
                        )
                        
                        if parsed_data and parsed_data.get('relevance_score', 0) > 0.2:
                            # Add crawl metadata
                            metadata = crawler.metadata_store.get(url, {})
                            parsed_data['crawl_metadata'] = metadata
                            
                            # Add technical content
                            tech_chunks = [
                                chunk for chunk in chunks 
                                if chunk.get('type') in ['code', 'table']
                                and chunk.get('relevance', 0) > 0.3
                            ]
                            if tech_chunks:
                                parsed_data['technical_content'] = tech_chunks
                                
                            structured_data.append(parsed_data)
                            self.logger.info(f"Successfully processed {url} with score {parsed_data['relevance_score']}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {e}")
                    continue
            
            # Sort by relevance
            structured_data.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Save output if we found any data
            if structured_data:
                self._save_output(structured_data, output_file)
                self.logger.info(f"Successfully processed {len(structured_data)} documents")
            else:
                self.logger.warning("No relevant content found")
                
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            return []

    def _prepare_output_file(self, output_file: str = None) -> str:
        """
        Prepare the output file path, creating directories as needed.
        """
        if output_file is None:
            output_folder = 'outputs'
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return os.path.join(output_folder, f'documents_{timestamp}.json')
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_file

    def _save_output(self, data: List[Dict[str, Any]], output_file: str = None):
        """Save the structured data to a JSON file with better error handling."""
        try:
            if not data:
                self.logger.warning("No data to save")
                return
                
            output_file = self._prepare_output_file(output_file)
            
            # Clean data for JSON serialization
            clean_data = []
            for doc in data:
                try:
                    # Convert numpy values to Python types
                    doc_copy = doc.copy()
                    if 'relevance_score' in doc_copy:
                        doc_copy['relevance_score'] = float(doc_copy['relevance_score'])
                    clean_data.append(doc_copy)
                except Exception as e:
                    self.logger.error(f"Error cleaning document for JSON: {e}")
                    continue
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Data saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")