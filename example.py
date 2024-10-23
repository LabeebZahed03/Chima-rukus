from rufus.client import RufusClient
import json

client = RufusClient()
url = 'https://www.taniarascia.com'
instructions = 'Find tutorials and articles about web development, JavaScript, and React.'

data = client.scrape(url, instructions, depth=2)

# Save data to JSON file
with open('taniarascia_data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Data saved to taniarascia_data.json")