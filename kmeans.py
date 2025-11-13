import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# (optional) from sklearn.decomposition import PCA

# 1) Load data
data = pd.read_csv("sales_data_sample.csv", encoding="ISO-8859-1")

# 2) Quick peek
print(data.head())

# 3) Missing values report
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# 4) Select numeric features only (copy to avoid chained assignment warnings)
features = data.select_dtypes(include=[np.number]).copy()

# Optional: drop obviously non-informative IDs (uncomment if present)
# features = features.drop(columns=["CustomerID", "OrderNumber"], errors="ignore")

# 5) Handle missing values (numeric columns)
features = features.fillna(features.mean(numeric_only=True))

# 6) Scale features (very important for K-Means)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)  # ndarray shape: (n_samples, n_features)

# 7) Elbow method to choose k
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.xticks(list(K))
plt.grid(True)
plt.savefig('op.png')
plt.show()

# 8) Pick k from the elbow (example: 3 — change based on your elbow)
optimal_k = 3

# 9) Fit final model and assign cluster labels
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled_features)
data["Cluster"] = labels

print("\nCluster counts:")
print(data["Cluster"].value_counts().sort_index())

# 10) (Optional) 2D visualization: uses first two standardized features
# Better: reduce with PCA to 2D for clearer plots
# pca = PCA(n_components=2, random_state=42)
# points_2d = pca.fit_transform(scaled_features)
points_2d = scaled_features[:, :2]

plt.figure(figsize=(8, 5))
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap="viridis", s=25)
plt.title(f"K-Means Clustering (k={optimal_k})")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.savefig('op1.png')
plt.show()
