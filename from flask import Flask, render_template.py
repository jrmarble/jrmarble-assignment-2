from flask import Flask, render_template, jsonify, request
import numpy as np
import random

app = Flask(__name__)

# Generate random dataset for KMeans
@app.route('/generate_dataset')
def generate_dataset():
    np.random.seed(42)
    data = np.random.randn(300, 2) * 10  # Generate 300 random 2D points
    return jsonify(data.tolist())

# Placeholder route for KMeans algorithm logic
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    # This is where you'd implement the KMeans algorithm
    # For now, we return a mock result for visualization purposes
    centroids = np.random.randn(int(request.json['k']), 2) * 10
    return jsonify({
        "centroids": centroids.tolist()
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
