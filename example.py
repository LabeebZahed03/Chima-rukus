from rufus.client import RufusClient

client = RufusClient()
instructions = "We're making a chatbot for blogs and projects."
url = "https://www.taniarascia.com"
data = client.scrape(url, instructions, depth=2)
client.save_to_file(data, 'website_data.json')
print("Data saved to sfgov_data.json")