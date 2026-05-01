import matplotlib.pyplot as plt
import pandas as pd
import requests
import json

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from CollaborativeRecommender import collaborative_exp

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

    # pass our song into model
    scaled_feats = scaler.transform(song_info)
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


def clustering_rec(song_with_feats):
    all_recs = input_to_rec(song_with_feats)
    # return the first three
    print(f"{all_recs[0].strip()}\n{all_recs[1].strip()}\n{all_recs[2].strip()}")


# function takes in song name, artist name, and dataframe
# returns list of song features
def check_dataset_or_spotify(song_name: str, artist_name: str, df):
    cred_file = "credentials.json"
    with open(cred_file, 'r') as f:
        data = json.load(f)

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=data["CLIENT_ID"],
        client_secret=data["CLIENT_SECRET"],
        redirect_uri=data["REDIRECT_URI"],
        scope=data["SCOPE"]
    ))

    match = df[
        (df['name'].str.lower() == song_name.lower()) &
        (df['artists'].str.lower().str.contains(artist_name.lower()))
        ]

    if not match.empty:
        print(f"Found '{song_name.capitalize()}' in local dataset")
        return match[FEATURES].iloc[0].values.reshape(1, -1)
    else:
        print(f"Not found locally, searching ReccoBeats...")
        results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
        if not results['tracks']['items']:
            print(f"Song '{song_name}' not found anywhere.")
            return None

        track = results['tracks']['items'][0]
        track_id = track['id']
        print(f"Found '{track['name']}' by {track['artists'][0]['name']} on Spotify")

        response = requests.get(
            "https://api.reccobeats.com/v1/audio-features",
            params={"ids": track_id},
            headers={"Accept": "application/json"}
        )
        content = response.json().get("content", [])
        if not content:
            print("ReccoBeats returned no features for this song.")
            return None

        song_feats = pd.DataFrame([content[0]])[FEATURES].fillna(0).values
        return song_feats


def cosine_sim_recs(sn, song_w_feats, df, n=3):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[FEATURES])
    song_scaled = scaler.transform(song_w_feats)

    similarities = cosine_similarity(song_scaled, df_scaled)[0]
    top_indices = similarities.argsort()[::-1][:n + 20]
    count = 0
    for idx in top_indices:
        if idx >= len(df):
            continue
        candidate = df.iloc[idx]['name']
        if candidate.lower() == sn.lower():
            continue
        artist = df.iloc[idx]['artists']
        print(f"{candidate} by {artist[2:-2]}")
        count += 1
        if count == n:
            break


if __name__ == "__main__":
    done = False
    X, df = load_data("data/tracks_features.csv")

    while not done:
        s = input("Enter song name: ")
        a = input("Enter artist name: ")
        print()
        print(f"Recommendations for {s.capitalize()} by {a.capitalize()}...")
        song_features = check_dataset_or_spotify(s, a, df)

        if song_features.any:
            print()
            clustering_rec(song_features)
            cosine_sim_recs(s, song_features, df)
            collaborative_exp(s, a, n=3)

        print("==" * 30)
        cont = input("Continue? (y/n): ")
        if cont != "y":
            done = True