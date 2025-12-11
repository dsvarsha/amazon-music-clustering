# silhouette_sample.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

print("Loading data (sampled)...")
df = pd.read_csv("single_genre_artists.csv")

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_ms"
]

# take a random sample to speed up silhouette computation
SAMPLE_N = 10000
if len(df) > SAMPLE_N:
    df_sample = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
else:
    df_sample = df.copy()

X = df_sample[audio_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 11)
results = []
out_path = "silhouette_sample_results.txt"
with open(out_path, "w") as f:
    f.write("k,silhouette_score\n")
    for k in k_range:
        print(f"Computing k={k} ...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}, silhouette={score:.6f}")
        f.write(f"{k},{score:.6f}\n")
        results.append((k, score))

print("Done. Results saved to", os.path.abspath(out_path))
print(results)
