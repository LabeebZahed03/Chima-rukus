# Rufus

Rufus is an AI-powered tool designed to intelligently crawl websites and extract relevant data for use in Retrieval-Augmented Generation (RAG) pipelines.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Initializing the Rufus Client](#initializing-the-rufus-client)
  - [Scraping a Website](#scraping-a-website)
- [How Rufus Works](#how-rufus-works)
  - [Crawler](#crawler)
  - [Parser](#parser)
- [Integrating Rufus into a RAG Pipeline](#integrating-rufus-into-a-rag-pipeline)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Intelligent web crawling** based on user instructions.
- **Advanced Natural Language Processing (NLP)** using spaCy for keyword extraction and content relevance.
- Extracts **metadata** such as titles, headings, and last updated dates.
- **Structured output** suitable for integration into RAG pipelines.

## Installation
### Prerequisites
- Python 3.7 or higher
- Google Chrome (for Selenium WebDriver)

### Install Rufus
```bash
pip install rufus-ai
```

### Install spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

## Usage
### Initializing the Rufus Client
```python
from rufus import RufusClient
import os

# Get your API key (currently any non-empty string)
key = os.getenv('RUFUS_API_KEY', 'your_default_api_key')

# Initialize Rufus client
client = RufusClient(api_key=key)
```

### Scraping a Website
```python
# Define the URL and instructions
url = 'https://example.com'
instructions = "Find information about product features and customer FAQs."

# Scrape the website
documents = client.scrape(url, instructions)

# Output the results
import json
with open('data.json', 'w') as f:
    json.dump(documents, f, indent=4)

print("Data saved to data.json")
```

## How Rufus Works
Rufus consists of two main components:

### Crawler
- Navigates through the provided website URL.
- Uses Selenium WebDriver to handle dynamic content and JavaScript-rendered pages.
- Collects HTML content from pages relevant to the user's instructions.

### Parser
- Processes the HTML content using BeautifulSoup.
- Utilizes spaCy's NLP capabilities to extract keywords from user instructions.
- Identifies and extracts relevant content based on the extracted keywords.
- Extracts metadata such as titles, headings, and last updated dates.

### Keyword Extraction and Content Relevance
- **Keyword Extraction:**
  - Uses spaCy's `en_core_web_lg` model for advanced NER and NLP tasks.
  - Extracts noun chunks, named entities, and significant nouns/proper nouns from the instructions.
- **Content Matching:**
  - Tokenizes and lemmatizes page content.
  - Matches content against extracted keywords to determine relevance.

## Integrating Rufus into a RAG Pipeline
To integrate Rufus into a Retrieval-Augmented Generation (RAG) pipeline:

1. **Data Collection:**
   - Use Rufus to scrape and parse relevant documents from target websites based on specific instructions.
2. **Data Preprocessing:**
   - Clean and preprocess the extracted data as required by your application (e.g., remove duplicates, handle special characters).
3. **Indexing:**
   - Feed the processed data into a vector store or database (e.g., Elasticsearch, Pinecone) to enable efficient retrieval.
4. **Retrieval:**
   - When a query is made, retrieve relevant documents from the vector store based on semantic similarity.
5. **Generation:**
   - Use a language model (e.g., GPT-3, GPT-4) to generate responses that are augmented with the retrieved documents.
6. **Feedback Loop:**
   - Optionally, use user feedback to further refine the retrieval and generation process.

## Dependencies
- BeautifulSoup4: HTML parsing
- Requests: Handling HTTP requests
- spaCy: Advanced NLP tasks
  - Requires `en_core_web_lg` language model
- Selenium: Web browser automation
- webdriver-manager: Manages WebDriver binaries

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License
This project is licensed under the MIT License.
