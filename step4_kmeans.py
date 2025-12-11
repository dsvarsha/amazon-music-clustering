# step4_kmeans.py (verbose, saves plots & scores)
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

print("STEP4: Starting K-Means script...")

# load
df = pd.read_csv("single_genre_artists.csv")
print(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} cols")

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_ms"
]

# check features exist
missing = [f for f in audio_features if f not in df.columns]
if missing:
    raise SystemExit(f"ERROR: Missing audio features: {missing}")

X = df[audio_features]
print("Selected audio features for clustering:", audio_features)
print("Preview (first row):")
print(X.iloc[0].to_dict())

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling done. Scaled shape:", X_scaled.shape)

# run elbow
inertia_list = []
K_range = range(2, 11)
print("Computing inertia for k in", list(K_range))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)
print("Inertia computed.")

# plot + save elbow
plt.figure(figsize=(8,5))
plt.plot(list(K_range), inertia_list, marker='o')
plt.title("Elbow Method - Inertia vs K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
elbow_path = "elbow_method.png"
plt.savefig(elbow_path, bbox_inches="tight")
print(f"Elbow figure saved -> {os.path.abspath(elbow_path)}")
plt.close()

# silhouette scores
silhouette_scores = []
scores_path = "silhouette_scores.txt"
with open(scores_path, "w") as f:
    f.write("k, silhouette_score\n")
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        line = f"{k}, {score:.6f}"
        f.write(line + "\n")
        print("Computed silhouette ->", line)
print(f"Silhouette scores written -> {os.path.abspath(scores_path)}")

# optionally plot silhouette (scores vs k) and save
plt.figure(figsize=(6,4))
plt.plot(list(K_range), silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
sil_plot = "silhouette_scores.png"
plt.savefig(sil_plot, bbox_inches="tight")
print(f"Silhouette plot saved -> {os.path.abspath(sil_plot)}")
plt.close()

# choose k (use best from silhouette or elbow; default 5)
best_k = 5
print("Using best_k =", best_k, "(you can change this after reviewing plots/scores)")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
print("KMeans fitted and cluster labels assigned.")

out_csv = "clustered_songs.csv"
df.to_csv(out_csv, index=False)
print(f"Clustered CSV saved -> {os.path.abspath(out_csv)}")

print("STEP4: Completed successfully.")
