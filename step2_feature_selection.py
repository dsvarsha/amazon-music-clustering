import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("single_genre_artists.csv")

# Features to keep for clustering
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

print("\n--- Selected Audio Features ---")
print(audio_features)

# Select only the audio feature columns
X = df[audio_features]

# Scale the features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Scaled Data Shape ---")
print(X_scaled.shape)

print("\nStep 2 completed — Feature Selection & Scaling DONE.")
