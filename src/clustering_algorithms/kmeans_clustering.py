import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
import json

def find_elbow_point(wcss):
    n = len(wcss)
    x1, y1 = 1, wcss[0]
    xN, yN = n, wcss[-1]

    max_distance = 0
    elbow_point = 1

    for i in range(2, len(wcss)):
        x0, y0 = i + 1, wcss[i]
        numerator = abs((yN - y1) * x0 - (xN - x1) * y0 + xN * y1 - yN * x1)
        denominator = ((yN - y1)**2 + (xN - x1)**2)**0.5
        distance = numerator / denominator

        if distance > max_distance:
            max_distance = distance
            elbow_point = i + 1
            
    return elbow_point

# Function to fit K-means
def fit_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    kmeans.fit(X)
    return kmeans

# Function to assign clusters to data
def assign_clusters_to_data(df, kmeans_model):
    df['Cluster'] = kmeans_model.labels_
    return df

# Function to save centroids to JSON file
def save_centroids(kmeans_model, filename='centroids.json'):
    centroids = kmeans_model.cluster_centers_.tolist()
    with open(filename, 'w') as f:
        json.dump(centroids, f)

# Load data
def load_data_from_file(filename):
    return pd.read_csv(filename)

if __name__ == '__main__':
    # Configuration parameters (you can move this to a separate config file)
    config = {
        'data_file': '../../data/processed_data/segmented_customers.csv',
        'max_clusters': 51
    }

    # Load data
    df = load_data_from_file(config['data_file'])

    # Selecting features for clustering
    X = df[['Age', 'Income', 'SpendingScore']]

    # Finding the elbow point using WCSS
    wcss = []
    for i in range(1, config['max_clusters']):
        kmeans = fit_kmeans(X, i)
        wcss.append(kmeans.inertia_)

  
    # Find the optimal number of clusters
    optimal_clusters = find_elbow_point(wcss)
    print(f"Optimal number of clusters: {optimal_clusters}")


    # Plotting the elbow method
    plt.plot(range(2, config['max_clusters']+1), wcss)
    plt.scatter(optimal_clusters, wcss[optimal_clusters-1], color='red', label=f'Elbow Point: {optimal_clusters}')
    plt.annotate(f'Elbow Point: {optimal_clusters}', (optimal_clusters, wcss[optimal_clusters-1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.legend()
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('../../plots/optimal_clusters.png', dpi=300)
    plt.show()

    

    # Fitting K-means with the optimal number of clusters
    kmeans_model = fit_kmeans(X, optimal_clusters)

    # Assign clusters to original data and save it
    df = assign_clusters_to_data(df, kmeans_model)
    df.to_csv('../../data/processed_data/segmented_customers_with_clusters.csv', index=False)

    # Save K-means model
    joblib.dump(kmeans_model, 'kmeans_model.pkl')

    # Save centroids
    save_centroids(kmeans_model)

    # Check the number of data points in clusters 2 and 5
cluster_2_count = df[df['Cluster'] == 2].shape[0]
cluster_5_count = df[df['Cluster'] == 5].shape[0]

print(f"Number of data points in cluster: {cluster_2_count}, {cluster_5_count}")
