import numpy as np
from flask import Flask, jsonify, request, session, render_template

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed if you plan to use sessions

# Generate random dataset
@app.route('/generate_dataset')
def generate_dataset():
    np.random.seed(42)
    data = np.random.randn(300, 2) * 10  # Generate 300 random 2D points
    return jsonify(data.tolist())

# Helper function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KMeans algorithm step-by-step
@app.route('/run_kmeans_step', methods=['POST'])
def run_kmeans_step():
    data = np.array(request.json['data'])  # Dataset from frontend
    k = int(request.json['k'])  # Number of clusters
    init_method = request.json['initMethod']  # Initialization method
    current_step = int(request.json['currentStep'])  # Current step

    # Initialize centroids on the first step
    if current_step == 0:
        if init_method == 'random':
            centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        elif init_method == 'farthest':
            centroids = farthest_first_initialization(data, k)
        elif init_method == 'kmeans++':
            centroids = kmeans_plus_plus_initialization(data, k)
        else:
            return jsonify({"error": "Invalid initialization method"})
        # Store initial centroids in session
        session['centroids'] = centroids.tolist()
    else:
        # Use stored centroids from previous step
        centroids = np.array(session['centroids'])

    # Perform one iteration of KMeans (assignment and update)
    clusters = assign_to_clusters(data, centroids)
    new_centroids = update_centroids(data, clusters, k)

    # Check if the centroids have converged (small distance threshold)
    converged = np.all(np.linalg.norm(new_centroids - centroids, axis=1) < 1e-4)

    # Update the stored centroids
    session['centroids'] = new_centroids.tolist()

    return jsonify({
        "centroids": new_centroids.tolist(),
        "clusters": clusters.tolist(),
        "converged": converged
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

# Farthest First Initialization (if needed)
def farthest_first_initialization(data, k):
    centroids = [data[np.random.randint(len(data))]]  # Choose a random point as the first centroid
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])
        next_centroid = data[np.argmax(distances)]
        centroids.append(next_centroid)
    return np.array(centroids)

# KMeans++ Initialization (if needed)
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
