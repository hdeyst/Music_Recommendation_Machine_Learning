import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TODO: make this not hard coded
K_VAL_MIN = 3
K_VAL_MAX = 10


def prepare_data():
    df = pd.read_csv("data/Music_Info.csv")

    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna()

    # drop categorical cols
    df = df.drop(columns=['track_id', 'name', 'artist', 'spotify_preview_url', 'spotify_id', 'tags', 'genre'])

    data = df.iloc[1:, :]
    feature_names = df.columns.tolist()
    return df, data, feature_names

def choose_k(scaled_x):
    # calculate optimal num clusters
    errors = []
    for i in range(K_VAL_MIN, K_VAL_MAX + 1):
        kmeans = KMeans(n_clusters=i, init="random", n_init=10, random_state=64)
        kmeans.fit_predict(scaled_x)
        errors.append(kmeans.inertia_)

    best_kval = K_VAL_MIN + int(np.argmax(errors))
    kvals = list(range(K_VAL_MIN, K_VAL_MAX + 1))

    plt.xticks(kvals)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum Squared Error")
    plt.plot(kvals, errors)
    plt.savefig("figures/choose_k_value.png")
    plt.show()

    print("The best k-value is: ", best_kval)
    return best_kval


def build_kmeans():
    df, X, features = prepare_data()
    print(X)

    # scale features so each has equal weight
    scaled_X = StandardScaler().fit_transform(X)
    print(f"{len(features)} features: {features}")
    print(f"Scaled data {scaled_X.shape}: \n{scaled_X}")

    # compute clustering w/ kmeans
    k_val = choose_k(scaled_X)
    kmeans = KMeans(init="random", n_clusters=k_val, n_init=10, random_state=64)
    kmeans.fit(scaled_X)
    k_means_cluster_centers = kmeans.cluster_centers_
    k_means_labels = pairwise_distances_argmin(scaled_X, k_means_cluster_centers)

    # reduce dimensions for graphing
    pca = PCA(n_components=2, random_state=64)
    coords = pca.fit_transform(scaled_X)
    print(coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps.get_cmap("tab20")

    for c in range(k_val):
        mask = k_means_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], color=colors(c), label=f"cluster {c}")

    plt.show()
    plt.savefig("figures/kmeans_clusters.png")


build_kmeans()




