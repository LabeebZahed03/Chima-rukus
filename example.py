from rufus.client import RufusClient
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
#Research papers used:
#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459555 (hybrid conditional model)
#Entity Resolution in Open-domain Conversations (NEL and NER)

def main():
    # Initialize client
    client = RufusClient(api_key='development_key')
    
    url = 'https://www.taniarascia.com'
    instructions = 'Extract information about articles on react, web development and SQL'
    
    try:
        print(f"\nStarting scrape of {url}")
        print(f"Instructions: {instructions}\n")
        
        documents = client.scrape(url, instructions, depth=2)
        
        print(f"\nFound {len(documents)} relevant documents\n")
        
        if documents:
            for i, doc in enumerate(documents[:5], 1):  # Show only top 5 docs
                print(f"\nDocument {i}:")
                print(f"URL: {doc.get('url')}")
                print(f"Title: {doc.get('title', '')}")
                print(f"Relevance: {doc.get('relevance_score', 0):.2f}")
                
                # Show sample content
                if doc.get('content'):
                    print("\nContent Sample:")
                    for chunk in doc['content'][:2]:  # Show only first 2 chunks
                        print(f"- {chunk.get('content', '')[:200]}...")
                
                # Show technical content
                if doc.get('technical_content'):
                    print("\nTechnical Content Sample:")
                    for chunk in doc['technical_content'][:2]:
                        print(f"Type: {chunk.get('type', 'unknown')}")
                        print(f"- {chunk.get('content', '')[:200]}...")
                        
                print("\n" + "="*80)  # Separator
                sys.stdout.flush()  # Ensure output is shown immediately
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nError during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()