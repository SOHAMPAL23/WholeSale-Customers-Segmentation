import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Create ml directory if it doesn't exist
os.makedirs('ml', exist_ok=True)

# Load the dataset
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

# Prepare data for clustering (exclude Channel and Region)
columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[columns]
print("\nFeatures for clustering:")
print(X.head())

# Check for any NaN values in the features
print("\nNaN values in features:")
print(X.isnull().sum())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)
print("Mean of scaled data (should be ~0):", np.mean(X_scaled, axis=0))
print("Std of scaled data (should be ~1):", np.std(X_scaled, axis=0))

# Apply k-Means clustering with k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = cluster_labels

print("\nk-Means clustering completed.")
print(f"Number of clusters: {optimal_k}")

# Save the trained models
import pickle

# Save k-Means model
with open('ml/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Save scaler
with open('ml/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels saved successfully:")
print("- ml/kmeans.pkl")
print("- ml/scaler.pkl")

print("\nScript completed successfully!")