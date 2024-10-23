# CHANGES THAT NEED TO BE MADE WHEN TRANSFERRED INTO PROD ENVIRONMENT FOR WEB APP:
# 1. REROUTE ALL .json FILE USAGE TO THE RELEVANT ONES THAT SERVED TO THE WEBSITE.
# 2. REROUTE ALL OUTPUT .JSON FILES, RANK PREDICTIONS, HIGH EV BETS, etc. TO CORRECT DATA STORAGAE DIRECTORIES.
# Import necessary libraries
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import getFile

# Just adding this, because I need to use directory a lot
odds_data_dir = '../odds_data/'

# 4. function
# Optimal bet function
def optimal_bet():
    # Load high_ev_bets_updated.json
    lhebu_dir = 'gradData/high_ev_bets_updated/'
    latest_high_ev_bets_updated = lhebu_dir + getFile.get_latest_creation_date_file('gradData/high_ev_bets_updated/')
    with open(latest_high_ev_bets_updated, 'r') as f:
        high_ev_bets = json.load(f)
    # Load odds_data
    latest_odds_data = odds_data_dir + getFile.get_latest_creation_date_file('../odds_data/')
    with open(latest_odds_data, 'r') as f1:
        odds_data = json.load(f1)

    # Extracting the relevant data
    implied_prob_home = [bet['implied_prob_home'] for bet in high_ev_bets]
    implied_prob_away = [bet['implied_prob_away'] for bet in high_ev_bets]
    expected_values = [bet['expected_value'] for bet in high_ev_bets]

    # Combine home and away implied probabilities and corresponding expected values
    implied_probabilities = implied_prob_home + implied_prob_away
    expected_values_combined = expected_values * 2 # Because we combine both home and away bets

    # Convert lists to numpy arrays for analysis
    implied_probabilities = np.array(implied_probabilities)
    expected_values_combined = np.array(expected_values_combined)

    # Perform a polynomial fit (degree 2) to find the optimal point (maximum) of the curve
    coefficients = np.polyfit(implied_probabilities, expected_values_combined, 2)
    polynomial = np.poly1d(coefficients)

    # Find the vertex of the parabola (the maximum point of the exepcted value curve)
    optimal_probability = -coefficients[1] / (2 * coefficients[0])
    optimal_expected_value = polynomial(optimal_probability)    

    def figure():
        # Plotting the curve with the optimal point
        implied_prob_range = np.linspace(0, 1, 100)
        expected_value_fitted = polynomial(implied_prob_range)

        plt.figure(figsize=(10,6))
        plt.plot(implied_probabilities, expected_values_combined, 'o',label='Data Points')
        plt.plot(implied_prob_range, expected_value_fitted, '-', label='Fitted Curve')
        plt.plot(optimal_probability, optimal_expected_value, 'ro', 
                 label=f'Optimal Point: {optimal_probability:.2f}, {optimal_expected_value:.2f}')
        plt.title('Implied Probability vs Expected Value with Optimal Bet Point')
        plt.xlabel('Implied Probability')
        plt.ylabel('Expected Value')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    # Matching the optimal probability with match infor from odds_data.json
    def find_closest_match(optimal_probability, odds_data):
        closest_match = None
        smallest_diff = float('inf')
        
        for match in odds_data:
            for bookmaker in match['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h': # Only interested in H2H market
                        for outcome in market['outcomes']:
                            # Calculate the implied probability from odds
                            implied_prob = 1 / outcome['price']
                            diff =abs(implied_prob - optimal_probability)

                            if diff < smallest_diff:
                                smallest_diff = diff 
                                closest_match = {
                                    'home_team': match['home_team'],
                                    'away_team': match['away_team'],
                                    'bookmaker': bookmaker['title'],
                                    'implied_probability': implied_prob,
                                    'odds': outcome['price'],
                                    'commence_time': match['commence_time']
                                }
            if closest_match:
                break # Break out once the closest match is found                    
        
        optimal_bet_dir = 'gradData/optimal_bet/'
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        optimal_bet_filename = optimal_bet_dir + current_date + '.json'
        # Write to a .CSV file
        with open(optimal_bet_filename, 'w') as outfile:
            json.dump({
                'Optimal Probability' : round(optimal_probability, 4),
                'Optimal Expected Value': round(optimal_expected_value, 4),
                'Closest Match': closest_match 
            }, outfile,indent=4)
        
        print(f"[UPDATED]:{optimal_bet_filename}")

        return closest_match
    
    find_closest_match(optimal_probability, odds_data)

# 2. function
# the matching model for matching high expected value bets with match commence times.
def matchingModel():
    # Load high EV bets and odds data
    high_ev_bets_updated_DIR = 'gradData/high_ev_bets_updated/'
    latest_high_ev_bets_updated = high_ev_bets_updated_DIR + getFile.get_latest_creation_date_file('gradData/high_ev_bets_updated/')
    with open(latest_high_ev_bets_updated, 'r') as f:
        high_ev_bets = json.load(f)
    
    latest_odds_data = odds_data_dir + getFile.get_latest_creation_date_file('../odds_data/')
    with open (latest_odds_data, 'r') as f:
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
    high_ev_bets_updated_dir = 'gradData/high_ev_bets_updated/'
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    high_ev_bets_updated_name = high_ev_bets_updated_dir + current_date + '.json'
    with open(high_ev_bets_updated_name, 'w') as f:
        json.dump(high_ev_bets, f, indent=4)
    
    print(f"[UPDATED]:{high_ev_bets_updated_name}")

# 3. function
# Small model for matching the commence times with the ranked predictions dataset.
def matchRankedPred():
    # Load the rank predictions, high EV bets, and odds data
    lrp_dir = 'gradData/rank_predictions/'
    latest_rank_predictions = lrp_dir + getFile.get_latest_creation_date_file('gradData/rank_predictions/')
    with open(latest_rank_predictions, 'r') as f:
        rank_predictions = json.load(f)
    
    lhebu_dir = 'gradData/high_ev_bets_updated/'
    latest_high_ev_bets_updated = lhebu_dir + getFile.get_latest_creation_date_file('gradData/high_ev_bets_updated/')
    with open(latest_high_ev_bets_updated, 'r') as f:
        high_ev_bets = json.load(f)

    latest_odds_data = odds_data_dir + getFile.get_latest_creation_date_file('../odds_data/')
    with open(latest_odds_data, 'r') as f:
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
                    'rank_prediction': prediction,
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
    
    rank_predictions_dir = 'gradData/rank_predictions_updated/'
    currrent_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    rank_predictions_filename = rank_predictions_dir + currrent_date + '.json'
    with open(rank_predictions_filename, 'w') as f:
        json.dump(rank_predictions, f, indent=4)
    
    print(f"[UPDATED]:{rank_predictions_filename}")

# 1. First function
def GradBoosted():
    # Load the dataset from the odds_data.json file
    latest_oddsFile = odds_data_dir + getFile.get_latest_creation_date_file('../odds_data/')
    with open(latest_oddsFile) as f:
        odds_data = json.load(f)
    
    # Extract relevant information from the JSON data
    matches = []
    for match in odds_data:
        for bookmaker in match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    home_team_price = next((outcome['price'] for outcome in outcomes if outcome['name'] == match['home_team']), None)
                    away_team_price = next((outcome['price'] for outcome in outcomes if outcome['name'] == match['away_team']), None)
                    if home_team_price is not None and away_team_price is not None:
                        matches.append({
                            'home_team': match['home_team'],
                            'away_team': match['away_team'],
                            'home_team_price': home_team_price,
                            'away_team_price': away_team_price,
                            'outcome': 1 if home_team_price < away_team_price else 0  # Assume home team wins if odds are lower
                        })
    df = pd.DataFrame(matches) # Convert to DataFrame
    X = df[['home_team_price', 'away_team_price']] # Feature columns and target column
    y = df['outcome']

    ### XGBCLASSIFIER MODEL SECTION
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model =xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss') # Setting up the XGBoost model
    model.fit(X_train, y_train) # train the XGBClassifier
    y_pred_proba = model.predict_proba(X_test) # Predict probas on test set

    # Evaluate model performance
    loss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Log Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Identify value bets, assume bookmaker's implied prob is calc from their odds
    X_test = X_test.copy() # Avoid SettingWithCopyWarning
    X_test['implied_prob_home'] = 1 / X_test['home_team_price']
    X_test['implied_prob_away'] = 1 / X_test['away_team_price']
    X_test['model_prob'] = y_pred_proba[:, 1] # Probability of home win (label 1)

    # Identify value bets where model probability > implied probability for the home team
    X_test['value_bet'] = X_test['model_prob'] > X_test['implied_prob_home']

    # Display the value bets - PRINTS
    value_bets = X_test[X_test['value_bet']]
    print("Value bets Identified:")
    print(value_bets[['home_team_price', 'away_team_price', 'model_prob', 'implied_prob_home']])

    ### XGBRANKER SECTION 
    # Group by 'home_team' to ensure multiple matches per group
    df['group'] = df['home_team'].factorize()[0]  # Assign group IDs based on unique teams

    # Print the dataframe to verify the group assignment
    print("Dataframe after group assignment:")
    print(df.head())

    # Prepare data for ranking
    X_rank = df[['home_team_price', 'away_team_price']]
    y_rank = df['outcome']
    groups = df['group']

    # Split the data into training and testing sets, including groups
    X_rank_train, X_rank_test, y_rank_train, y_rank_test, groups_train, groups_test = train_test_split(
        X_rank, y_rank, groups, test_size=0.2, random_state=42
    )

    # Create group information for training and testing
    def get_group_sizes(groups):
        group_sizes = groups.value_counts().sort_index().tolist()
        return group_sizes

    # Ensure group sizes add up properly for training and testing
    groups_train_sizes = get_group_sizes(groups_train)
    groups_test_sizes = get_group_sizes(groups_test)

    # Verify group sizes to match number of rows
    assert sum(groups_train_sizes) == len(X_rank_train), "Group sizes for training do not match the number of training samples."
    assert sum(groups_test_sizes) == len(X_rank_test), "Group sizes for testing do not match the number of testing samples."

    # Set up the XGBRanker model
    ranker = xgb.XGBRanker(
        objective='rank:pairwise',
        eval_metric='ndcg',
        learning_rate=0.1,
        gamma=1.0,
        min_child_weight=0.1,
        max_depth=6
    )

    # Train the ranker model
    ranker.fit(
        X_rank_train,
        y_rank_train,
        group=groups_train_sizes,
        eval_set=[(X_rank_test, y_rank_test)],
        eval_group=[groups_test_sizes],
        verbose=True
    )

    # Predict ranks on the test set
    rank_predictions = ranker.predict(X_rank_test)

    # Check if predictions were made
    if rank_predictions is not None and len(rank_predictions) == len(X_rank_test):
        # Create a copy of the test data for ranking purposes 
        X_rank_test = X_rank_test.copy()

        # Introduce some derived features to add more variability to rank calculations
        if 'implied_prob' in X_rank_test.columns and 'model_prob' in X_rank_test.columns:
            # Calculate the difference between the model probability and implied probability 
            X_rank_test['prob_diff'] = X_rank_test['model_prob'] - X_rank_test['implied_prob']

            # Add this difference to rank predictions to modify the ranks
            rank_predictions = rank_predictions + X_rank_test['prob_diff']
        
        # Scale the rank predictions for better differentiation
        rank_predictions_scaled = rank_predictions / rank_predictions.max()

        # Add slight random noise for differentiation
        random_noise = np.random.normal(0, 0.1, len(rank_predictions_scaled))
        rank_predictions_noisy = rank_predictions_scaled + random_noise

        # Clip the rank predictions to ensure they are within reasonable range
        rank_predictions_noisy = np.clip(rank_predictions_noisy, -5, 5)

        # Assign the noisy rank predictions back to the data
        X_rank_test['rank_prediction'] = rank_predictions_noisy
    else:
        print("Rank predictions were not made correctly or the length does not match the test set.")

    # Prepare data to be written to a JSON file
    data_to_write = {
        "rank_predictions": rank_predictions.tolist(),
        "group_sizes_train": groups_train_sizes,
        "group_sizes_test": groups_test_sizes
    }

    # WRITE OUT THE DATA .json file
    # Define the path and name of the output file
    rank_predictions_dir = 'gradData/rank_predictions/'
    new_date = datetime.now().strftime('%Y-%m-%d-%H_%M')
    output_file_path = rank_predictions_dir + new_date + '.json'

    # Write the data to a JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data_to_write, file, indent=4)

    print(f"[UPDATED]: {output_file_path}")

    # EXTRA FUNCTIONALITY: INTERPRETATION AND VISUALIZATION
    # 1. Calculate Expected Value (EV) for Each bet 
    def calculate_expected_value(row):
        return (row['model_prob'] * row['home_team_price']) - (1 - row['model_prob'])
    
    X_test['expected_value'] = X_test.apply(calculate_expected_value, axis=1)

    def scatter_plot():
        # Visualize value bets vs. Non-value bets
        plt.figure(figsize=(10,6))
        plt.scatter(X_test['home_team_price'], X_test['model_prob'], c=X_test['value_bet'], cmap='bwr',
                    alpha=0.6)
        plt.xlabel('Expected Value')
        plt.ylabel('Model Probability of Home Win')
        plt.title('Value Bets (Red) vs. Non-Value bets (Blue)')
        plt.show()

    def histogram():
        # Histogram of Expected Values
        plt.figure(figsize=(10,6))
        plt.hist(X_test['expected_value'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Expected Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Expected Values for Bets')
        plt.show()

    high_ev_bets = X_test[X_test['expected_value'] > 1.0] # Bets wth high expected value

    ### DATASET OUTPUT - for HIGH EV BETS
    # Convert high_ev_bets to a dictionary format
    high_ev_bets_dict = high_ev_bets.to_dict(orient='records')
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    high_ev_bets_dir = 'gradData/high_ev_bets/'
    filename_high_ev_bets = high_ev_bets_dir + current_date + '.json'
    with open(filename_high_ev_bets, 'w') as json_file:
        json.dump(high_ev_bets_dict, json_file, indent=4)
    
    print(f"[UPDATED HIGH_EV_BETS]: {filename_high_ev_bets}")

    def update_highEVbetsData():
        # Load the odds data
        with open(latest_oddsFile) as f:
            odds_data = json.load(f)
        
        # Load high_ev_bets
        latest_high_ev_bets = getFile.get_latest_creation_date_file('gradData/high_ev_bets/')
        # Ensure the full path is used
        high_ev_dir = 'gradData/high_ev_bets/'
        latest_high_ev_bets_path = high_ev_bets_dir + latest_high_ev_bets
        with open(latest_high_ev_bets_path) as f:
            high_ev_bets = json.load(f)
        
        # Create a mapping from home and away prices to match information
        match_mapping = {}
        for match in odds_data:
            home_team = match['home_team']
            away_team = match['away_team']
            for bookmaker in match['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        outcomes = market['outcomes']
                        home_price = next((outcome['price'] for outcome in outcomes if outcome['name'] == home_team), None)
                        away_price = next((outcome['price'] for outcome in outcomes if outcome['name'] == away_team), None)
                        if home_price and away_price:
                            match_mapping[(home_price, away_price)] = {
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker': bookmaker['title']
                            }
        # Update high_ev_bets with match information
        for bet in high_ev_bets:
            home_price = bet['home_team_price']
            away_price = bet['away_team_price']
            match_info = match_mapping.get((home_price, away_price))
            if match_info:
                bet.update(match_info)
        
        # Save updated high_ev_bets
        high_ev_bets_updated_dir = 'gradData/high_ev_bets_updated/'
        high_ev_bets_updated_name = high_ev_bets_updated_dir + current_date + '.json'
        with open(high_ev_bets_updated_name, 'w') as f:
            json.dump(high_ev_bets, f, indent=4)
        
        print(f"[UPDATED]: {high_ev_bets_updated_name}")
    
    update_highEVbetsData() # running the function

GradBoosted() # First run the gradboosted model
matchAlgo_highEVbets = matchingModel() # Secondly merge the commence times of the match to the updated expected values.
matchAlgo_rankedPred = matchRankedPred() # Match the commence and matches with the ranked predictions.
getOptimalBetModel = optimal_bet() # Uses the optimal_bet() to calculate the most optimal

