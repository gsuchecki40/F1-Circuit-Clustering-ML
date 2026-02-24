from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('FinalMergedWithCircuits.csv')
agg = df.groupby(['circuit_id', 'circuit_name']).median(numeric_only=True).reset_index()

features = ['track_length_km','num_corners','num_slow_corners','num_high_speed_corners',
            'elevation_change_m','full_throttle_pct','avg_speed_kmh','longest_straight_m',
            'track_width_m','altitude_m','typical_pit_stops','surface_abrasiveness',
            'overtaking_difficulty','tyre_degradation']

X = agg[features].astype(float)

X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

pca = PCA(n_components= min(10, X.shape[1]),random_state=42).fit_transform(X)
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca)

kmeans = KMeans(n_clusters=6, random_state=42).fit_predict(pca)
hdb = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(pca)

agg['umap_x'], agg['umap_y'] = emb[:, 0], emb[:, 1]
agg['kmeans'], agg['hdbscan'] = kmeans, hdb
agg.to_csv('circuit_clusters.csv', index=False)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=agg, x='umap_x', y='umap_y', hue='hdbscan', palette='tab10', s=60)
plt.savefig('circuit_clusters_umap.png',dpi=200)
plt.close()