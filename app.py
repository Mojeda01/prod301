import json
import os
from flask import Flask, jsonify, render_template


app = Flask(__name__)

# Route to serve the odds as JSON
@app.route('/odds')
def odds_endpoint():
    try:
        odds_dir = 'odds_data'
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
