import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from api_song_loading import FEATURES, get_tracks_info, get_top_spotify_tracks

# TODO: make this not hard coded
K_VAL = 10
NUM_NEIGHBORS = 5

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates(subset=['id'])
    df = df.dropna()

    # extract data w/o labels
    X = df[FEATURES].values
    return X, df

def train_kmeans(X, scaler):
    scaled_X = scaler.fit_transform(X)

    # create clusters w/ kmeans
    km = KMeans(n_clusters=K_VAL, init="random", n_init=10, random_state=64)
    clusters = km.fit(scaled_X)

    # get neighbors w/ knn & euclidean distance
    knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric="euclidean")
    knn.fit(scaled_X)

    return scaler, km, knn, clusters, scaled_X

def recommend():
    scaler = StandardScaler()

    X, df = load_data("data/tracks_features.csv")
    scaler, km, knn, clusters, scaled_X = train_kmeans(X, scaler)

    df['cluster'] = clusters

    top_tracks = get_top_spotify_tracks()
    track_features = get_tracks_info(top_tracks)

    scaled_feats = scaler.fit_transform(track_features)
    distances, idxs = knn.kneighbors(scaled_feats)

    recs = df.iloc[idxs[0]]
    print(recs)

recommend()