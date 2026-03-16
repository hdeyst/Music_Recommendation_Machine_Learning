import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.pyplot import tight_layout
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TODO: make this not hard coded
K_VAL_MIN = 2
K_VAL_MAX = 20


def prepare_data():
    df = pd.read_csv("data/Music_Info.csv")

    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna()

    # drop categorical cols
    feature_df = df.drop(columns=['track_id', 'name', 'artist', 'spotify_preview_url', 'spotify_id', 'tags', 'genre'])
    song_df = df[['track_id', 'name', 'artist', 'spotify_preview_url', 'spotify_id', 'tags', 'genre']]

    data = feature_df.iloc[1:, :]
    feature_names = feature_df.columns.tolist()
    return feature_df, song_df, data, feature_names

def choose_k(scaled_x):
    # calculate optimal num clusters
    inertias = []
    for i in range(K_VAL_MIN, K_VAL_MAX + 1):
        kmeans = KMeans(n_clusters=i, init="random", n_init=10, random_state=64)
        kmeans.fit_predict(scaled_x)
        inertias.append(kmeans.inertia_)

    best_kval = K_VAL_MIN + int(np.argmax(inertias))
    kvals = list(range(K_VAL_MIN, K_VAL_MAX + 1))

    plt.xticks(kvals)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum Squared Error")
    plt.plot(kvals, inertias)
    plt.tight_layout()
    plt.savefig("figures/choose_k_value.png")
    plt.show()

    print("The best k-value is: ", best_kval)
    return best_kval


def build_kmeans(df, X):
    # scale features so each has equal weight
    scaled_X = StandardScaler().fit_transform(X)
    print(f"Scaled data {scaled_X.shape}: \n{scaled_X}")

    # TODO: k_val = choose_k(scaled_X)
    k_val = 3
    kmeans = KMeans(init="random", n_clusters=k_val, n_init=10, random_state=64)
    kmeans.fit(scaled_X)
    k_means_cluster_centers = kmeans.cluster_centers_
    k_means_labels = pairwise_distances_argmin(scaled_X, k_means_cluster_centers)
    # df['cluster'] = kmeans.labels_
    print(len(k_means_labels))
    print(len(kmeans.labels_))

    # reduce dimensions for graphing
    pca = PCA(n_components=2, random_state=64)
    coords = pca.fit_transform(scaled_X)
    print(coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps.get_cmap("tab20")

    for c in range(k_val):
        mask = k_means_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], color=colors(c), label=f"cluster {c}")

    plt.savefig("figures/kmeans_clusters.png")
    plt.show()

    return df, scaled_X


def rec_engine(song_title, df, scaled_X):
    song_idx = df[df["name"] == song_title].index[0]
    cluster = df.loc[song_idx, ["cluster"]]
    print(cluster)


def main():
    feature_df, song_df, data, feature_names = prepare_data()
    print(f"{len(feature_names)} features: {feature_names}")


    df, scaled_x = build_kmeans(feature_df, data)
    # rec_engine("Mr. Brightside", df, scaled_x)

main()