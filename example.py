from rufus.client import RufusClient

# Initialize the client (API key is optional for now)
client = RufusClient(api_key='your_api_key')

# Define instructions
instructions = "Extract the title of the page."

# URL to scrape
url = "https://www.taniarascia.com"

# Perform scraping
data = client.scrape(url, instructions)

# Output the results
print(data)