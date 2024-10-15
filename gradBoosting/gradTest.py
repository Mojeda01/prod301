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


def GradBoosted():
    # Load the dataset from the odds_data.json file
    with open('odds_data.json') as f:
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

    # Print rank predictions to verify
    print("Rank Predictions:")
    print(rank_predictions)

    # Check if predictions were made
    if rank_predictions is not None and len(rank_predictions) == len(X_rank_test):
        # Add the predictions to the test dataframe to see how they rank
        X_rank_test = X_rank_test.copy()
        X_rank_test['rank_prediction'] = rank_predictions

        # Display ranked bets
        print("Ranked Bets:")
        print(X_rank_test.sort_values(by='rank_prediction', ascending=False))
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
    output_file_path = "rank_predictions.json"

    # Write the data to a JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data_to_write, file, indent=4)

    print(f"Data written to {output_file_path}")

    # EXTRA FUNCTIONALITY: INTERPRETATION AND VISUALIZATION
    # 1. Calculate Expected Value (EV) for Each bet 
    def calculate_expected_value(row):
        return (row['model_prob'] * row['home_team_price']) - (1 - row['model_prob'])
    
    X_test['expected_value'] = X_test.apply(calculate_expected_value, axis=1)
    print("Expected value for each Bet:")
    print(X_test[['home_team_price', 'away_team_price', 'model_prob', 'expected_value']])

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
    print("High Expected Value Bets:")
    print(high_ev_bets[['home_team_price', 'away_team_price', 'model_prob', 'expected_value']])

    ### DATASET OUTPUT - for HIGH EV BETS
    # Convert high_ev_bets to a dictionary format
    high_ev_bets_dict = high_ev_bets.to_dict(orient='records')
    with open('high_ev_bets.json', 'w') as json_file:
        json.dump(high_ev_bets_dict, json_file, indent=4)
    
    print("[+++] HIGH EV bets have been saved to high_ev_bets.json [BEFORE MATCHING]")

    def update_highEVbetsData():
        # Load the odds dataa
        with open('odds_data.json') as f:
            odds_data = json.load(f)
        
        # Load high_ev_bets
        with open('high_ev_bets.json') as f:
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
        with open('high_ev_bets_updated.json', 'w') as f:
            json.dump(high_ev_bets, f, indent=4)
        
        print("[+++] Updated high_ev_bets.json with match information saved to high_ev_bets_updated.json")
    
    update_highEVbetsData() # running the function



GradBoosted()