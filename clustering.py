import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TODO: make this not hard coded
K_VAL = 3


def prepare_data():
    df = pd.read_csv("data/Music_Info.csv")

    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna()

    # drop categorical cols
    df = df.drop(columns=['track_id', 'name', 'artist', 'spotify_preview_url', 'spotify_id', 'tags', 'genre'])

    data = df.iloc[1:, :]
    feature_names = df.columns.tolist()
    return df, data, feature_names


def build_kmeans():
    df, X, features = prepare_data()
    print(X)

    # scale features so each has equal weight
    scaled_X = StandardScaler().fit_transform(X)
    print(f"{len(features)} features: {features}")
    print(f"Scaled data {scaled_X.shape}: \n{scaled_X}")

    # compute clustering w/ kmeans
    kmeans = KMeans(init="random", n_clusters=K_VAL, n_init=10, random_state=64)
    kmeans.fit(scaled_X)
    k_means_cluster_centers = kmeans.cluster_centers_
    k_means_labels = pairwise_distances_argmin(scaled_X, k_means_cluster_centers)

    # reduce dimensions for graphing
    pca = PCA(n_components=2, random_state=64)
    coords = pca.fit_transform(scaled_X)
    print(coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps.get_cmap("tab20")

    for c in range(K_VAL):
        mask = k_means_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], color=colors(c), label=f"cluster {c}")

    plt.show()





build_kmeans()




