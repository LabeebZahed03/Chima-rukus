def calculate_text_relevance(text: str, keywords: List[str], entities: Dict[str, List[str]]) -> float:
    """
    Calculate relevance score for text based on keywords and entities.
    
    :param text: Text to analyze
    :param keywords: List of keywords
    :param entities: Dictionary of entity types and values
    :return: Relevance score between 0 and 1
    """
    if not text:
        return 0.0
        
    score = 0.0
    
    # Keyword matching
    for keyword in keywords:
        if keyword.lower() in text.lower():
            score += 0.3
            
    # Entity matching
    for entity_type, entity_values in entities.items():
        for entity in entity_values:
            if entity.lower() in text.lower():
                score += 0.2
                
    return min(1.0, score)

def extract_main_content(html: str) -> str:
    """
    Extract main content from HTML, removing boilerplate.
    
    :param html: HTML content
    :return: Main content as string
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
        
    # Try to find main content container
    main_content = None
    for container in ['main', 'article', 'div[@role="main"]']:
        main_content = soup.find(container)
        if main_content:
            break
            
    return main_content.get_text(strip=True) if main_content else soup.get_text(strip=True)

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    :param text: Text to clean
    :return: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\-.,?!]', '', text)
    return text.strip()