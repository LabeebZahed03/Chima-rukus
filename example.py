from rufus.client import RufusClient
import os
import json

# Set the API key (currently any non-empty string)
key = os.getenv('RUFUS_API_KEY', 'default_key')

# Initialize Rufus client
client = RufusClient(api_key=key)

url = 'https://www.taniarascia.com'
instructions = "extract articles about javascript, react, web-development"

# Scrape the website
documents = client.scrape(url, instructions)

# Output the results
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'testwebsite.json')

with open(file_path, 'w') as f:
    json.dump(documents, f, indent=4)

print(f"Data saved to {file_path}")
