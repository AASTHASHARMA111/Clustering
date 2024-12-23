import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the data from the Excel files
train_file_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/train.xlsx'
test_file_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/test.xlsx

# Read the train and test datasets
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Preprocess the train data
X_train = train_data.drop(columns=['target'])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming 3 clusters for this example
kmeans.fit(X_train_scaled)

# Predict clusters for the train data
train_data['predicted_cluster'] = kmeans.labels_

# Preprocess the test data using the same scaler
X_test = test_data
X_test_scaled = scaler.transform(X_test)

# Predict clusters for the test data
test_data['predicted_cluster'] = kmeans.predict(X_test_scaled)

# Extract the cluster centroids
centroids = kmeans.cluster_centers_

# Transform the centroids back to the original scale
centroids_original_scale = scaler.inverse_transform(centroids)

# Create a DataFrame for the centroids for better visualization
centroids_df = pd.DataFrame(centroids_original_scale, columns=X_train.columns)

# Display the centroids
print(centroids_df)

# Save the resulting test data with predicted clusters
test_data.to_excel('test_with_clusters.xlsx', index=False)
