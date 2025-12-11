# 🎶 Amazon Music Clustering using Unsupervised Machine Learning  
### 📌 A Data Science Project by Varsha S  
Unsupervised ML | K-Means | PCA | Data Visualization | Music Analytics

---

## 📍 Project Overview  
With millions of tracks on platforms like Amazon Music, manually categorizing songs into genres or moods is nearly impossible.  
This project uses **Unsupervised Machine Learning (K-Means Clustering)** to automatically group songs based on their **audio characteristics**, revealing natural musical clusters such as:

- 🔊 Energetic / Dance tracks  
- 🎧 Acoustic / Chill songs  
- 🗣️ Speech-heavy vocal tracks  

The goal is to show how audio features can be used to understand music similarity without needing labels.

---

## 🎯 Problem Statement  
The aim of this project is to **cluster Amazon Music songs** using numerical audio features like **tempo, energy, danceability, loudness, valence**, etc., and identify meaningful patterns in the dataset.

Clustering helps streaming platforms with:  
✔ Personalized playlist generation  
✔ Better music recommendations  
✔ Identifying listener preferences  
✔ Mood/genre discovery automation  

---

## 📂 Dataset  
The dataset (`single_genre_artists.csv`) includes **95,837 songs** with 23 columns such as:  
- `danceability`, `energy`, `loudness`, `speechiness`,  
- `acousticness`, `instrumentalness`, `tempo`, `valence`,  
- metadata like artist name, song ID, release date, etc.

Only the **audio features** were used for clustering.

---

## 🧠 Project Workflow  
### **1️⃣ Data Exploration (EDA)**  
- Checked missing values (none)  
- Verified data types and distributions  
- Identified 10 key audio features for clustering  

### **2️⃣ Preprocessing & Feature Scaling**  
Selected features:  
danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness,
valence, tempo, duration_ms
Scaled data using `StandardScaler`.

### **3️⃣ PCA Visualization**  
Applied **PCA (2 components)** to visualize high-dimensional audio data in 2D space.

### **4️⃣ Choosing Best K (Elbow + Silhouette)**  
- Elbow Method suggested k ≈ 3–5  
- Silhouette Score was highest at **k = 3**  
➡️ Final model used **3 clusters**

### **5️⃣ K-Means Clustering**  
Performed K-Means with `k = 3` and added cluster labels to dataset.

### **6️⃣ Cluster Profiling & Visualization**  
Created visualizations:  
- Elbow curve  
- Silhouette scores  
- PCA scatter plot  
- Cluster mean bar chart  
- Radar chart  
- Feature boxplots  
- Tempo distributions  

---

## 🔍 Cluster Interpretation  
Based on feature profiles, the model identified **3 natural clusters**:

### 🎤 **Cluster 0 – Speech-Heavy / Vocal-Rich Tracks**  
- Highest speechiness  
- Medium energy & danceability  
- Shorter duration  
💡 *Represents rap-like, talk-heavy, spoken-word style music*

### 🎸 **Cluster 1 – Acoustic / Calm / Emotional Songs**  
- Highest acousticness  
- Lowest energy  
- Softer mood (low valence)  
💡 *Represents chill, soft, emotional, acoustic tracks*

### ⚡ **Cluster 2 – Energetic / Happy Dance-Pop Tracks**  
- Highest energy  
- Loudest songs  
- Fast tempo  
- Highest valence (happy mood)  
💡 *Represents upbeat, energetic, dance-style tracks*

---

## 📊 Visualizations Included  
All graphs are saved in the repository:  
- `elbow_method.png`  
- `silhouette_scores.png`  
- `pca_scatter_labeled.png`  
- `cluster_means_bar.png`  
- `cluster_means_radar.png`  
- `feature_boxplots.png`  
- `tempo_hist_by_cluster.png`

---

## 📁 Repository Structure  
📦 amazon-music-clustering
├── step1_eda.py
├── step2_feature_selection.py
├── step3_pca.py
├── step4_kmeans.py
├── step5_cluster_analysis.py
├── step6_visualizations.py
├── clustered_songs.csv
├── cluster_profiles.csv
├── top_songs_per_cluster.csv
├── elbow_method.png
├── pca_scatter_labeled.png
├── cluster_means_bar.png
├── cluster_means_radar.png
├── feature_boxplots.png
└── tempo_hist_by_cluster.png


---

## 🚀 How to Run the Project  
1. Clone the repository:  

---

## 🚀 How to Run the Project  
1. Clone the repository:  

---

## 🚀 How to Run the Project  
1. Clone the repository:  
git clone https://github.com/dsvarsha/amazon-music-clustering.git

2. Install required packages:  
pip install pandas numpy scikit-learn matplotlib seaborn


3. Run clustering scripts:  
python step1_eda.py
python step2_feature_selection.py
python step3_pca.py
python step4_kmeans.py
python step5_cluster_analysis.py
python step6_visualizations.py


---

## 📌 Conclusion  
This project successfully demonstrates how unsupervised ML can uncover hidden music patterns using audio features. It provides valuable insights for:

- Music recommendation systems  
- Mood-based playlist creation  
- Artist/song similarity discovery  
- Audio-based segmentation  

---

## ✨ Author  
**Varsha Suresh**  
📍 Data Science & Machine Learning Enthusiast  
📧 varshasuresh0708@gmail.com  
🔗 GitHub: https://github.com/dsvarsha  

