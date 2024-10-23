from rufus.client import RufusClient
import os

# api key -(currently any non-empty string)
key = os.getenv('RUFUS_API_KEY', 'your_default_api_key')

# Initialize Rufus client
client = RufusClient(api_key=key)

url = 'https://www.taniarascia.com'
instructions = "extract articles about javascript, react, web-development"

# Scrape the website
documents = client.scrape(url, instructions)

# Output the results
output_folder = 'outputs'
file_path = os.path.join(output_folder, 'testwebsite.json')
import json
with open(file_path, 'w') as f:
    json.dump(documents, f, indent=4)

print(f"Data saved to {file_path}")