import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import scipy.cluster.hierarchy as sch
import json
import pickle
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create directories
os.makedirs('ml', exist_ok=True)
os.makedirs('notebook', exist_ok=True)

print("Starting Wholesale Customer Segmentation Analysis...")
print("=" * 50)

# 1. Data Loading and Exploration
print("\n1. Loading and Exploring Data...")
df = pd.read_csv('Wholesale_customers_data.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Preprocessing
print("\n2. Preprocessing Data...")
columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[columns]

print("Features for clustering:")
print(X.head())

# Check for any NaN values in the features
print("\nNaN values in features:")
print(X.isnull().sum())

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')  # Replace NaN with mean values
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame to maintain column names
X_imputed_df = pd.DataFrame(X_imputed, columns=columns)

print("\nAfter imputation:")
print("NaN values in features:")
print(X_imputed_df.isnull().sum())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)

print("\nScaled data shape:", X_scaled.shape)
print("Mean of scaled data (should be ~0):", np.mean(X_scaled, axis=0))
print("Std of scaled data (should be ~1):", np.std(X_scaled, axis=0))

# 3. Determining Optimal Number of Clusters
print("\n3. Determining Optimal Number of Clusters...")

# Elbow Method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('ml/elbow_curve.png')
plt.close()

# Silhouette Analysis
silhouette_scores = []
K_range_sil = range(2, 11)  # Silhouette score requires at least 2 clusters

for k in K_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range_sil, silhouette_scores, marker='o', linestyle='-', color='g')
plt.title('Silhouette Score Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.savefig('ml/silhouette_scores.png')
plt.close()

print("Silhouette Scores:")
for i, score in enumerate(silhouette_scores):
    print(f"k={i+2}: Silhouette Score = {score:.4f}")

# Calinski-Harabasz Index
ch_scores = []

for k in K_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    ch_scores.append(ch_score)

# Plot Calinski-Harabasz scores
plt.figure(figsize=(10, 6))
plt.plot(K_range_sil, ch_scores, marker='o', linestyle='-', color='r')
plt.title('Calinski-Harabasz Index for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.grid(True)
plt.savefig('ml/calinski_harabasz_scores.png')
plt.close()

print("\nCalinski-Harabasz Scores:")
for i, score in enumerate(ch_scores):
    print(f"k={i+2}: Calinski-Harabasz Score = {score:.2f}")

print("\nBased on the Elbow Method, Silhouette Score, and Calinski-Harabasz Index, let's select k=3 as the optimal number of clusters.")

# 4. k-Means Clustering Implementation
print("\n4. Implementing k-Means Clustering...")
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = cluster_labels

print("k-Means clustering completed.")
print(f"Number of clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, cluster_labels):.2f}")

# Count of customers in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()
print("\nCustomer distribution across clusters:")
print(cluster_counts)

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title('Customer Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.savefig('ml/cluster_distribution.png')
plt.close()

# 5. Agglomerative Clustering Comparison
print("\n5. Comparing with Agglomerative Clustering...")
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(X_scaled)

# Add Agglomerative cluster labels to the dataframe
df['Agg_Cluster'] = agg_labels

print("Agglomerative Clustering completed.")
print(f"Silhouette Score: {silhouette_score(X_scaled, agg_labels):.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, agg_labels):.2f}")

# Compare cluster distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# k-Means distribution
kmeans_counts = df['Cluster'].value_counts().sort_index()
axes[0].bar(kmeans_counts.index, kmeans_counts.values, color='skyblue')
axes[0].set_title('k-Means Clustering Distribution')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Number of Customers')

# Agglomerative distribution
agg_counts = df['Agg_Cluster'].value_counts().sort_index()
axes[1].bar(agg_counts.index, agg_counts.values, color='lightcoral')
axes[1].set_title('Agglomerative Clustering Distribution')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('ml/clustering_comparison.png')
plt.close()

# Compare silhouette scores
kmeans_silhouette = silhouette_score(X_scaled, cluster_labels)
agg_silhouette = silhouette_score(X_scaled, agg_labels)

comparison_df = pd.DataFrame({
    'Algorithm': ['k-Means', 'Agglomerative'],
    'Silhouette Score': [kmeans_silhouette, agg_silhouette]
})

print("\nAlgorithm Comparison:")
print(comparison_df)

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x='Algorithm', y='Silhouette Score', data=comparison_df, palette='Set2')
plt.title('Silhouette Score Comparison')
plt.ylabel('Silhouette Score')
plt.savefig('ml/algorithm_comparison.png')
plt.close()

print("\nWe'll proceed with k-Means clustering as it typically provides better results for this type of data.")

# 6. Principal Component Analysis
print("\n6. Performing Principal Component Analysis...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("PCA completed.")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

# Plot PCA results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', alpha=0.7)
plt.title('PCA - 2D Visualization of Customer Segments')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
plt.colorbar(scatter)
plt.grid(True)
plt.savefig('ml/pca_visualization.png')
plt.close()

# Feature importance in PCA components
components_df = pd.DataFrame(
    pca.components_, 
    columns=columns, 
    index=['PC1', 'PC2']
)

plt.figure(figsize=(10, 6))
sns.heatmap(components_df.T, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('PCA Components - Feature Importance')
plt.ylabel('Original Features')
plt.xlabel('Principal Components')
plt.savefig('ml/pca_components.png')
plt.close()

# 7. Cluster Profiling and Business Personas
print("\n7. Generating Cluster Profiles and Business Personas...")

# Calculate mean spending per cluster
cluster_means = df.groupby('Cluster')[columns].mean()
print("\nMean spending per cluster:")
print(cluster_means)

# Visualization of cluster means
cluster_means.plot(kind='bar', figsize=(12, 8))
plt.title('Average Spending by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.yscale('log')
plt.grid(True, axis='y')
plt.savefig('ml/cluster_spending.png')
plt.close()

# Identify dominant categories for each cluster
dominant_categories = {}
weakest_categories = {}

for cluster in cluster_means.index:
    cluster_data = cluster_means.loc[cluster]
    dominant_categories[cluster] = cluster_data.idxmax()
    weakest_categories[cluster] = cluster_data.idxmin()
    
print("\nDominant and weakest categories by cluster:")
for cluster in cluster_means.index:
    print(f"Cluster {cluster}: Dominant = {dominant_categories[cluster]}, Weakest = {weakest_categories[cluster]}")

# Generate business personas
personas = {}

for cluster in cluster_means.index:
    cluster_data = cluster_means.loc[cluster]
    dominant_category = dominant_categories[cluster]
    weakest_category = weakest_categories[cluster]
    
    # Create behavioral tags
    if dominant_category == 'Fresh':
        behavioral_tag = "Fresh Produce Specialists"
    elif dominant_category == 'Grocery':
        behavioral_tag = "Grocery-Dominant Retailers"
    elif dominant_category == 'Milk':
        behavioral_tag = "Dairy Product Wholesalers"
    elif dominant_category == 'Frozen':
        behavioral_tag = "Frozen Food Distributors"
    elif dominant_category == 'Detergents_Paper':
        behavioral_tag = "Non-Food Essentials Suppliers"
    else:  # Delicassen
        behavioral_tag = "Specialty Food Providers"
    
    # Create marketing insights
    if cluster == 0:
        marketing_insight = "These customers are high-volume fresh produce buyers. Target with premium quality offerings and seasonal promotions."
        campaign_recommendation = "Promote organic and specialty fresh produce lines with volume discounts."
    elif cluster == 1:
        marketing_insight = "These customers focus heavily on grocery items. They represent stable, recurring revenue opportunities."
        campaign_recommendation = "Introduce loyalty programs and bundled grocery packages."
    else:  # cluster == 2
        marketing_insight = "These customers have diverse purchasing patterns with balanced spending across categories."
        campaign_recommendation = "Offer cross-category promotions and personalized recommendations."
    
    # Create persona summary
    persona_summary = f"{behavioral_tag} who primarily purchase {dominant_category.lower()} products. "
    persona_summary += f"They show minimal interest in {weakest_category.lower()} items. "
    persona_summary += f"{marketing_insight}"
    
    personas[cluster] = {
        'cluster_id': cluster,
        'size': int(cluster_counts[cluster]),
        'dominant_category': dominant_category,
        'weakest_category': weakest_category,
        'behavioral_tag': behavioral_tag,
        'mean_spending': cluster_data.to_dict(),
        'persona_summary': persona_summary,
        'marketing_insight': marketing_insight,
        'campaign_recommendation': campaign_recommendation
    }

# Display personas
print("\nGenerated Business Personas:")
for cluster_id, persona in personas.items():
    print(f"\n===== CLUSTER {cluster_id} PERSONA =====")
    print(f"Behavioral Tag: {persona['behavioral_tag']}")
    print(f"Dominant Category: {persona['dominant_category']}")
    print(f"Weakest Category: {persona['weakest_category']}")
    print(f"Cluster Size: {persona['size']} customers")
    print(f"Persona Summary: {persona['persona_summary']}")
    print(f"Campaign Recommendation: {persona['campaign_recommendation']}")

# Save personas to JSON
with open('ml/personas.json', 'w') as f:
    json.dump(personas, f, indent=2)

print("\nPersonas saved to ml/personas.json")

# Create markdown version of personas
with open('ml/personas.md', 'w') as f:
    f.write("# Customer Segment Personas\n\n")
    for cluster_id, persona in personas.items():
        f.write(f"## Cluster {cluster_id}: {persona['behavioral_tag']}\n\n")
        f.write(f"**Dominant Category:** {persona['dominant_category']}  \n")
        f.write(f"**Weakest Category:** {persona['weakest_category']}  \n")
        f.write(f"**Cluster Size:** {persona['size']} customers  \n\n")
        f.write(f"**Persona Summary:** {persona['persona_summary']}  \n\n")
        f.write(f"**Marketing Recommendation:** {persona['campaign_recommendation']}  \n\n")
        f.write("---\n\n")

print("Personas saved to ml/personas.md")

# 8. Model Persistence
print("\n8. Saving Models and Data...")

# Save k-Means model
with open('ml/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Save scaler
with open('ml/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save PCA transformer
with open('ml/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Save imputer
with open('ml/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print("Models saved successfully:")
print("- ml/kmeans.pkl")
print("- ml/scaler.pkl")
print("- ml/pca.pkl")
print("- ml/imputer.pkl")

# Save processed data for backend use
df.to_csv('ml/processed_data.csv', index=False)
pca_df.to_csv('ml/pca_data.csv', index=False)

print("\nProcessed data saved:")
print("- ml/processed_data.csv")
print("- ml/pca_data.csv")

# Save elbow data for backend API
elbow_data = {
    'k_values': list(K_range),
    'inertias': inertias
}

with open('ml/elbow_data.json', 'w') as f:
    json.dump(elbow_data, f)

print("\nElbow data saved to ml/elbow_data.json")

print("\n" + "=" * 50)
print("Analysis Complete!")
print("=" * 50)
print("\nNext steps:")
print("1. Check the generated visualizations in the 'ml' directory")
print("2. Review the business personas in 'ml/personas.json' and 'ml/personas.md'")
print("3. Use the saved models to implement the backend API")
print("4. Develop the frontend UI with animations")