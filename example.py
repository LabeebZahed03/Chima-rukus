# example.py

from rufus.client import RufusClient

client = RufusClient()
instructions = "Find tutorials and articles about web development, JavaScript, and React."
url = "https://www.taniarascia.com"
data = client.scrape(url, instructions, depth=2)
client.save_to_file(data, 'taniarascia_data.json')
print("Data saved to taniarascia_data.json")
