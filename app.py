import json
import os
from flask import Flask, jsonify, render_template, send_file, abort
from collections import defaultdict
from datetime import datetime
from gradBoosting.getFile import get_latest_creation_date_file

app = Flask(__name__)

# Utility function to fetch the latest high_ev_bets_updated file.
def fetch_latest_high_ev_bets_data():
    base_dir = os.path.join('gradBoosting', 'gradData', 'high_ev_bets_updated')
    latest_file = get_latest_creation_date_file(base_dir)

    if not latest_file:
        return None
    
    # Load the file content assuming it's a JSON or text file, adjust as necessary 
    with open(os.path.join(base_dir, latest_file), 'r') as file:
        data = json.load(file) # parse the JSON file 
    return data 

# Utility function to fetch the latest optimal_bet data
def fetch_latest_optimal_bet_data():
    base_dir = os.path.join('gradBoosting', 'gradData', 'optimal_bet')
    latest_file = get_latest_creation_date_file(base_dir)

    if not latest_file:
        return None
    
    # Load and parse the JSON file 
    with open(os.path.join(base_dir, latest_file), 'r') as file:
        data = json.load(file) # Parse the JSON file
    return data 


# Route to serve the betRanker page
@app.route('/betRanker', methods=['GET'])
def betRanker():
    # Fetch JSON data using the dedication functions
    high_ev_bets_data = fetch_latest_high_ev_bets_data()
    optimal_bet_data = fetch_latest_optimal_bet_data()

    if not high_ev_bets_data or not optimal_bet_data:
        return abort(404, description="Data files not found or empty")
    
    # Pass the JSON data to the template for rendering 
    return render_template('betRanker.html', high_ev_bets_data=high_ev_bets_data, optimal_bet_data=optimal_bet_data)

# Route to serve the odds as JSON
@app.route('/odds')
def odds_endpoint():
    try:
        odds_dir = os.path.join(os.path.dirname(__file__), 'odds_data')
        # latest_file = sorted(os.listdir(odds_dir))[-1] # Old latest_file var
        latest_file = max(os.listdir(odds_dir), key=lambda f: os.path.getmtime(os.path.join(odds_dir, f)))
        with open(os.path.join(odds_dir, latest_file), 'r') as f:
            odds_data = json.load(f)
        return jsonify(odds_data)
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
