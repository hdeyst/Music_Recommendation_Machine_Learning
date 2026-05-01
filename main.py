import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from CollaborativeRecommender import collaborative_exp
from cosine_sim import get_audio_features, cos_sim

import warnings
warnings.filterwarnings("ignore", message="")


FEATURES = [
    'danceability',
    'acousticness',
    'instrumentalness',
    'loudness'
]

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates(subset=['id'])
    df = df.dropna()

    # extract data w/o labels
    X = df[FEATURES].values
    return X, df

def plot_elbow(X, scaler):
    scaled_X = scaler.fit_transform(X)

    inertias = []
    ks = [i for i in range(1, 20)]
    for k in ks:
        km = KMeans(n_clusters=k, init="random", n_init=10)
        km.fit(scaled_X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True)
    plt.show()

def train_kmeans(X, scaler):
    # Uncomment to see elbow plot
    # plot_elbow(X, scaler)

    K_VAL = 5
    scaled_X = scaler.fit_transform(X)

    # create clusters w/ kmeans
    km = KMeans(n_clusters=K_VAL, init="random", n_init=10)
    clusters = km.fit(scaled_X)

    # get neighbors w/ knn & euclidean distance
    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(scaled_X)

    return scaler, km, knn, clusters, scaled_X


def input_to_rec(song_info):
    scaler = StandardScaler()

    X, df = load_data("data/tracks_features.csv")
    scaler, km, knn, clusters, scaled_X = train_kmeans(X, scaler)

    df['cluster'] = clusters.labels_

    # Uncomment to see PCA graph
    # X_2d = PCA(n_components=2).fit_transform(X)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters.labels_)
    # plt.show()

    # scale specified song features
    track_features_df = pd.DataFrame(song_info)[FEATURES]

    # pass our song into model
    scaled_feats = scaler.transform(track_features_df.values)
    distances, idxs = knn.kneighbors(scaled_feats)

    return format_recs(df.iloc[idxs[0]])


# takes in df of song recommendations and formats them nicely
def format_recs(rec_list):
    song_lst_formatted = []
    for i, rec in rec_list.iterrows():
        song_lst_formatted.append(one_rec_to_str(rec))

    return song_lst_formatted


def one_rec_to_str(song_info):
    artists_str = ""
    title_str = song_info['name']
    artist_lst = song_info['artists']
    artist_lst = artist_lst.replace("[", "").replace("]", "").replace("'", "")

    a_lst = list(artist_lst.split(","))

    for i, artist in enumerate(a_lst):
        artists_str += artist

        if i < len(a_lst) - 1:
            artists_str += ", "

    return f"\t{title_str.strip()} by {artists_str.strip()}"



def spotipy_connect():
    # get credentials from the json file
    cred_file = "credentials.json"
    with open(cred_file, 'r') as f:
        data = json.load(f)

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=data["CLIENT_ID"],
        client_secret=data["CLIENT_SECRET"],
        redirect_uri=data["REDIRECT_URI"],
        scope=data["SCOPE"]
    ))
    return sp


def call_spotipy(song_name, artist_name):
    retrieved_song = ""
    sid = ""
    sp = spotipy_connect()
    results = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
    if results['tracks']['items']:
        retrieved_song = results['tracks']['items'][0]
        sid = results['tracks']['items'][0]['id']

    return retrieved_song, sid

def clustering_rec(song, artist):
    song_w_id, sid = call_spotipy(song, artist)
    song_with_feats = get_audio_features([sid])

    if song_with_feats:
        all_recs = input_to_rec(song_with_feats)
    else:
        return False
    # return the first three
    print(f"{all_recs[0].strip()}\n{all_recs[1].strip()}\n{all_recs[2].strip()}")



if __name__ == "__main__":
    done = False

    while not done:
        s = input("Enter song name: ")
        a = input("Enter artist name: ")
        print()

        print(f"\nkmeans rec:")
        done = clustering_rec(s, a)
        print(f"\ncollaborative rec:")
        collaborative_exp(s, a, n=3)
        print(f"\ncosine similarity rec:")
        cos_sim(s, a)

        print("==" * 30)
        cont = input("Continue? (y/n): ")
        if cont != "y":
            done = True