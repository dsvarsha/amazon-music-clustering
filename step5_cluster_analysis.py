# step5_cluster_analysis.py
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("STEP5: Cluster profiling + PCA scatter starting...")

# Load original data
df = pd.read_csv("single_genre_artists.csv")
print("Loaded rows:", df.shape[0])

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_ms"
]

# Ensure features present
missing = [f for f in audio_features if f not in df.columns]
if missing:
    raise SystemExit("Missing features: " + str(missing))

X = df[audio_features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling done. Scaled shape:", X_scaled.shape)

# Choose k (set to 3 based on silhouette sampling)
best_k = 3
print("Using best_k =", best_k)

# Fit KMeans and assign labels
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels
print("Clusters assigned. Unique clusters:", np.unique(labels))

# Save clustered CSV
out_csv = "clustered_songs.csv"
df.to_csv(out_csv, index=False)
print("Saved clustered CSV ->", os.path.abspath(out_csv))

# Compute cluster centers in scaled space, inverse transform to original units
centers_scaled = km.cluster_centers_
centers_orig = scaler.inverse_transform(centers_scaled)
cluster_profiles = pd.DataFrame(centers_orig, columns=audio_features)
cluster_profiles["cluster"] = range(best_k)
cols = ["cluster"] + audio_features
cluster_profiles = cluster_profiles[cols]
profile_csv = "cluster_profiles.csv"
cluster_profiles.to_csv(profile_csv, index=False)
print("Saved cluster profiles ->", os.path.abspath(profile_csv))

# Top songs per cluster (by popularity_songs) - top 10 each
top_list = []
for c in range(best_k):
    sub = df[df["cluster"] == c].sort_values("popularity_songs", ascending=False)
    top = sub[["id_songs","name_song","name_artists","popularity_songs","cluster"]].head(10)
    top_list.append(top)
top_df = pd.concat(top_list, ignore_index=True)
top_csv = "top_songs_per_cluster.csv"
top_df.to_csv(top_csv, index=False)
print("Saved top songs per cluster ->", os.path.abspath(top_csv))

# PCA for visualization (2D)
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df["PC1"] = pca_coords[:, 0]
df["PC2"] = pca_coords[:, 1]

# scatter plot colored by cluster
plt.figure(figsize=(10,7))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']  # up to 5 palettes
for c in range(best_k):
    mask = df["cluster"] == c
    plt.scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"], s=6, alpha=0.6, label=f"Cluster {c}", color=colors[c % len(colors)])
plt.legend(markerscale=2)
plt.title(f"PCA Scatter Plot (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
scatter_img = f"pca_cluster_scatter_k{best_k}.png"
plt.savefig(scatter_img, bbox_inches="tight")
plt.close()
print("Saved PCA cluster scatter ->", os.path.abspath(scatter_img))

print("STEP5: Completed. Check these files in your project folder:")
print(" -", out_csv)
print(" -", profile_csv)
print(" -", top_csv)
print(" -", scatter_img)
