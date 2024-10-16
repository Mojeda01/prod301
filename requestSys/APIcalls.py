import json
import requests
import os
from datetime import datetime

def readKey():
    with open('APIKEY.md', 'r') as file:
        API_key = file.read().strip()
    
    return API_key

print(readKey())

def get_premier_league_odds(api_key, region='eu', market='h2h'):
    """
    Fetches Premier League odds and events using the Odds-API and exports the data to the odds_data directory.

    Parameters:
        api_key (str): Your Odds-API key
        region (str): The region to get odds for. Default is 'eu'.
        market (str): The betting market type. Default is 'h2h' (Head to Head).
    
    Returns:
        None
    """

    url = 'https://api.the-odds-api.com/v4/sports/soccer_epl/odds'
    params = {
        'apiKey': api_key,
        'regions': region,
        'markets': market,
        'oddsFormat': 'decimal'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        try:
            odds_data = response.json()
            print(json.dumps(odds_data, indent=4))
            os.makedirs('odds_data', exist_ok=True) # Create the odds_data directory if it doesn't exist
            timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M') # Generate filename using the current date and time
            filename = f'odds_data/{timestamp}_odds.json'

            with open(filename, 'w') as f: # Save the data to a JSON file in the odds_data directory
                json.dump(odds_data, f, indent=4)
        except json.JSONDecodeError:
            print('Error decoding JSON response')
    else:
        print(f'Error fetching data: {response.status_code}, {response.text}')


get_premier_league_odds(readKey())