import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import joblib


# Load the K-means model
kmeans_model = joblib.load('../clustering_algorithms/kmeans_model.pkl')

# Load the original data with cluster labels
df = pd.read_csv('../../data/processed_data/segmented_customers_with_clusters.csv')
df_raw = pd.read_csv('../../data/raw_data/customer_data.csv')


# Check if all clusters have data points
for cluster in range(kmeans_model.n_clusters):
    if not any(df['Cluster'] == cluster):
        print(f"Warning: No data points found for cluster {cluster}")

# Get the cluster centroids from the model
centroids = kmeans_model.cluster_centers_

# Define a function for creating faceted 2D scatter plots
def plot_faceted_2d(df,feature_x, feature_y, title):
    # Create a FacetGrid, one plot per cluster
    g = sns.FacetGrid(df, col='Cluster', col_wrap=4, height=3)
    
    # Map the scatter plot for each cluster
    g.map(plt.scatter, feature_x, feature_y, alpha=0.7)
    
    # Add titles and labels
    g.set_titles('Cluster {col_name}')
    g.set_axis_labels(feature_x, feature_y)
    
    # Add super title
    g.fig.suptitle(title, fontsize=16)
    g.fig.subplots_adjust(top=0.9)  # Adjust the placement of the title to make room for the super title

    
    # Save the plot
    save_path = f'../../plots/{title.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=300)
    
    # Show plot
    plt.show()

# Column indices for Age, Income, and SpendingScore
age_idx, income_idx, spending_idx = 0, 1, 2 


# Create faceted 2D plots for each pair
plot_faceted_2d(df,'Age', 'Income', 'Age vs Income')
plot_faceted_2d(df,'Age', 'SpendingScore', 'Age vs Spending Score')
plot_faceted_2d(df,'Income', 'SpendingScore', 'Income vs Spending Score')



# Create the 3D scatter plot
fig = go.Figure()

# Sort the unique cluster values
sorted_clusters = sorted(df['Cluster'].unique())

# Plot each cluster with a separate color and label, in sorted order
for cluster in sorted_clusters:
    cluster_data = df[df['Cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['Age'], y=cluster_data['Income'], z=cluster_data['SpendingScore'],
        mode='markers',
        marker=dict(size=5, opacity=0.6),
        name=f'Cluster {cluster}'
    ))

# Add centroids as distinct scatter points with text labels
for i, centroid in enumerate(centroids):
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
        mode='markers+text',
        marker=dict(color='black', size=10),
        text=f'Centroid {i}',
        hoverinfo='text',
        name=f'Centroid {i}'
    ))

fig.update_layout(title='3D Scatter Plot of Customer Segments', margin=dict(l=0, r=0, b=0, t=0))

fig.write_image("../../plots/3D_Scatter_Plot_of_Customer_Segments.png")

fig.show()


# Create a violin plot for each feature
features = ['Age', 'Income', 'SpendingScore']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Cluster', y=feature, data=df)
    plt.title(f'Violin Plot of {feature} by Cluster')
    plt.savefig(f'../../plots/Violin_Plot_{feature}_by_Cluster.png')
    plt.show()

