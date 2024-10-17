import json

# the matching model for matching high expected value bets with match commence times.
def matchingModel():
    # Load high EV bets and odds data
    with open('high_ev_bets_updated.json', 'r') as f:
        high_ev_bets = json.load(f)
    
    with open ('odds_data.json', 'r') as f:
        odds_data = json.load(f)

    # Create a mapping for commence_time based on teams
    commence_time_mapping = {}

    # Populate the mapping from odds_data
    for match in odds_data:
        home_team = match['home_team']
        away_team = match['away_team']
        commence_time = match['commence_time']
        # Use a tuple of teams as the key
        commence_time_mapping[(home_team, away_team)] = commence_time
    
    # Now, update high EV bets with the commence_time
    for bet in high_ev_bets:
        home_team = bet['home_team']
        away_team = bet['away_team']
        # Find the matching commence_time
        key = (home_team, away_team)
        if key in commence_time_mapping:
            bet['commence_time'] = commence_time_mapping[key]
        else:
            bet['commence_time'] = 'Unknown' # In case a match is not found.
    
    # Save the updated high EV bets back to the original file
    with open('high_ev_bets_updated.json', 'w') as f:
        json.dump(high_ev_bets, f, indent=4)
    
    print("[<3] high_ev_bets_updated.json successfully updated with commence_time")

# Small model for matching the commence times with the ranked predictions dataset.
def matchRankedPred():
    # Load the rank predictions, high EV bets, and odds data
    with open('rank_predictions.json', 'r') as f:
        rank_predictions = json.load(f)
    
    with open('high_ev_bets_updated.json', 'r') as f:
        high_ev_bets = json.load(f)

    with open('odds_data.json', 'r') as f:
        odds_data = json.load(f)

    # Create a mapping of commence_time and teams from odds_data.json
    commence_time_mapping = {}
    for match in odds_data:
        key = (match['home_team'], match['away_team'])
        commence_time_mapping[key] = {
            'commence_time': match['commence_time'],
            'home_team': match['home_team'],
            'away_team': match['away_team']
        }
    
    # Iterate through the rank predictions and add corresponding match info
    updated_rank_predictions = []

    for idx, prediction in enumerate(rank_predictions['rank_predictions']):
        # Use index or some pattern to correlate rank_predictions to teams
        if idx < len(high_ev_bets):
            home_team = high_ev_bets[idx]['home_team']
            away_team = high_ev_bets[idx]['away_team']
            key = (home_team, away_team)

            # Check if match information is available
            if key in commence_time_mapping:
                match_info = commence_time_mapping[key]
                updated_rank_predictions.append({
                    'rank_predictiopn': prediction,
                    'home_team': match_info['home_team'],
                    'away_team': match_info['away_team'],
                    'commence_time': match_info['commence_time']
                })
            else:
                # Handle missing match case
                updated_rank_predictions.append({
                    'rank_prediction': prediction,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': 'Unknown'
                })
    # Save the updated rank predictions back to the same file
    rank_predictions['rank_predictions'] = updated_rank_predictions

    with open('rank_predictions.json', 'w') as f:
        json.dump(rank_predictions, f, indent=4)
    
    print("rank_predictions.json successfully updated with match info.")