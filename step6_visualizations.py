# step6_visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid", font_scale=1.0)

print("STEP6: Generating visualizations...")

# Load files produced earlier
# If you used the scripts above, clustered_songs.csv and cluster_profiles.csv exist.
df = pd.read_csv("clustered_songs.csv")
profiles = pd.read_csv("cluster_profiles.csv")

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_ms"
]

# 1) Grouped bar chart: mean audio features per cluster
print(" -> Creating cluster means bar chart...")
means = df.groupby("cluster")[audio_features].mean().reset_index()
# Reorder columns to show the most interpretable features first
plot_features = ["danceability","energy","loudness","acousticness","valence","tempo","duration_ms"]

plt.figure(figsize=(12,6))
x = np.arange(len(means['cluster']))
width = 0.12
for i, feat in enumerate(plot_features):
    plt.bar(x + i*width, means[feat], width=width, label=feat)
plt.xticks(x + width*(len(plot_features)-1)/2, means['cluster'])
plt.xlabel("Cluster")
plt.ylabel("Mean value (original scale)")
plt.title("Cluster Mean Audio Features (grouped bar)")
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig("cluster_means_bar.png", bbox_inches="tight")
plt.close()
print("   saved -> cluster_means_bar.png")

# 2) Radar chart of normalized cluster profiles (nice summary)
print(" -> Creating radar chart...")
# normalize profile values 0-1 for radar (except loudness and duration we scale differently)
p = profiles.copy().set_index("cluster")
# For radar it's nicer to scale each feature between 0 and 1 across clusters
p_norm = (p - p.min()) / (p.max() - p.min())
categories = p_norm.columns.tolist()
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for i, idx in enumerate(p_norm.index):
    values = p_norm.loc[idx].tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=f"Cluster {idx}")
    ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_title("Normalized Cluster Profiles (Radar)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("cluster_means_radar.png", bbox_inches="tight")
plt.close()
print("   saved -> cluster_means_radar.png")

# 3) Feature boxplots for a few key features
print(" -> Creating boxplots for energy, danceability, loudness...")
features_box = ["energy", "danceability", "loudness"]
plt.figure(figsize=(12,4))
for i, feat in enumerate(features_box):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="cluster", y=feat, data=df, palette=colors)
    plt.title(f"{feat} by cluster")
plt.tight_layout()
plt.savefig("feature_boxplots.png", bbox_inches="tight")
plt.close()
print("   saved -> feature_boxplots.png")

# 4) Tempo histograms per cluster (subplots)
print(" -> Creating tempo histograms by cluster...")
unique_clusters = sorted(df["cluster"].unique())
n = len(unique_clusters)
cols = min(3, n)
rows = int(np.ceil(n/cols))
plt.figure(figsize=(12, 3*rows))
for i, c in enumerate(unique_clusters):
    plt.subplot(rows, cols, i+1)
    subset = df[df["cluster"] == c]
    sns.histplot(subset["tempo"], bins=30, kde=True)
    plt.title(f"Cluster {c} - tempo (n={len(subset)})")
    plt.xlabel("Tempo (BPM)")
plt.tight_layout()
plt.savefig("tempo_hist_by_cluster.png", bbox_inches="tight")
plt.close()
print("   saved -> tempo_hist_by_cluster.png")

# 5) PCA scatter (resave labelled, larger and prettier)
print(" -> Creating labeled PCA scatter plot...")
if "PC1" not in df.columns or "PC2" not in df.columns:
    # compute PCA quickly (we'll approximate for visualization)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df[audio_features].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df["PC1"] = coords[:,0]
    df["PC2"] = coords[:,1]

plt.figure(figsize=(10,7))
for i, c in enumerate(unique_clusters):
    mask = df["cluster"] == c
    plt.scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"], s=8, alpha=0.6, label=f"Cluster {c}", color=colors[i % len(colors)])
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot (clusters labelled)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scatter_labeled.png", bbox_inches="tight")
plt.close()
print("   saved -> pca_scatter_labeled.png")

print("STEP6: All visualizations created. Check the png files in your project folder.")
