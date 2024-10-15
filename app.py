import json
import os
from flask import Flask, jsonify, render_template
import datetime

app = Flask(__name__)

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the odds as JSON
@app.route('/odds')
def get_odds():
    try:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        odds_file = f'odds_data/{current_date}_odds.json'
        with open(odds_file, 'r') as f:
            odds = json.load(f)
        return jsonify(odds)
    except FileNotFoundError:
        return jsonify({"error": "Odds file not found"}), 404

@app.route('/events')
def get_events():
    try:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        events_file = f'odds_data/{current_date}_events.json'
        with open(events_file, 'r') as f:
            events = json.load(f)
        return jsonify(events)
    except FileNotFoundError:
        return jsonify({"error":"Events file not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
