import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Generate random dataset
@app.route('/generate_dataset')
def generate_dataset():
    np.random.seed(42)
    data = np.random.randn(300, 2) * 10  # Generate 300 random 2D points
    return jsonify(data.tolist())

# Helper function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KMeans algorithm implementation
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    data = np.array(request.json['data'])  # Dataset passed from frontend
    k = int(request.json['k'])  # Number of clusters
    init_method = request.json['initMethod']  # Initialization method
    
    # If manual centroids are provided, use them
    if init_method == 'manual':
        centroids = np.array(request.json['manualCentroids'])
    elif init_method == "random":
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    elif init_method == "farthest":
        centroids = farthest_first_initialization(data, k)
    elif init_method == "kmeans++":
        centroids = kmeans_plus_plus_initialization(data, k)
    else:
        return jsonify({"error": "Invalid initialization method"})

    # Main KMeans loop
    max_iterations = 100
    for _ in range(max_iterations):
        # Assignment step: assign each point to the nearest centroid
        clusters = assign_to_clusters(data, centroids)
        
        # Update step: calculate new centroids as mean of the points in each cluster
        new_centroids = update_centroids(data, clusters, k)
        
        # Check for convergence (if centroids don't change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return jsonify({
        "centroids": centroids.tolist(),
        "clusters": clusters.tolist()
    })

# Assign points to the nearest centroid
def assign_to_clusters(data, centroids):
    clusters = np.zeros(data.shape[0])
    for i, point in enumerate(data):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters[i] = np.argmin(distances)
    return clusters

# Update centroids by calculating the mean of all points assigned to each cluster
def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return new_centroids

# Farthest First Initialization
def farthest_first_initialization(data, k):
    centroids = [data[np.random.randint(len(data))]]  # Choose a random point as the first centroid
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])
        next_centroid = data[np.argmax(distances)]
        centroids.append(next_centroid)
    return np.array(centroids)

# KMeans++ Initialization
def kmeans_plus_plus_initialization(data, k):
    centroids = [data[np.random.randint(len(data))]]
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])
        probabilities = distances / distances.sum()
        next_centroid = data[np.random.choice(len(data), p=probabilities)]
        centroids.append(next_centroid)
    return np.array(centroids)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
