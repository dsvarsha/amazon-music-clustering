import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("single_genre_artists.csv")

# Audio features to use for PCA
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

# Select only the audio feature columns
X = df[audio_features]

# Scale before PCA (very important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (reduce to 2 dimensions)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

print("\n--- PCA Components Shape (should be (95837, 2)) ---")
print(pca_components.shape)

# Add PCA results to dataframe
df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]

# Save PCA output for future visualization
df.to_csv("pca_output.csv", index=False)

print("\nStep 3 completed — PCA DONE and saved to pca_output.csv")

