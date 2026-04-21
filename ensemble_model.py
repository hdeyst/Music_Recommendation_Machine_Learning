import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from scipy import sparse
from spotipy.oauth2 import SpotifyClientCredentials

N = 1
SONG = 'Washed'
ARTIST = 'The Jack Wharff Band'

### ------------- NICKS CODE ------------------ ###
# read in files
mat_songs_features = sparse.load_npz("data/mat_song_features.npz")
song_metadata = pd.read_csv('data/song_metadata.csv')




decode_id_song = {
    row['name']: i
    for i, row in enumerate(song_metadata.to_dict('records'))
}

decode_id_artists = {
    i: (row['name'], row['artist'])
    for i, row in enumerate(song_metadata.to_dict('records'))
}

decode_id_title_artist = {
    i: f"{title} - {artist}"
    for i, (title, artist) in decode_id_artists.items()
}
class Recommender: 
    '''
    Initializes the recommendation model
    metric: the distance metric we use (cosine)
    algorithm: algorithm used to compute the neasest neighbors (brute)
    n_neighbors: number of neighbors to use for queries (20) (very important)
    '''
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.model = self._recommender().fit(data)

    # initializes and fits the knn model
    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, 
                            n_neighbors=self.k, n_jobs=-1)
    
    # gets the _get_recommendations reccomendation and returns the song title with _map_indeces_to_song_title
    def make_recommendation(self, new_song, n_recommendations):
            recommendations = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
            return self._map_indeces_to_song_title(recommendation_ids=[idx for idx, _ in recommendations])

    def _get_recommendations(self, new_song, n_recommendations):
        recom_song_id = self._fuzzy_matching(song_artist_string=new_song)
        
        distances, indices = self.model.kneighbors(
            self.data[recom_song_id], 
            n_neighbors=n_recommendations+1
        )
        
        return sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), 
            key=lambda x: x[1]
        )[:0:-1]

    # mapps the index of the song to the song title
    def _map_indeces_to_song_title(self, recommendation_ids):
        # reverse_map = {v: k for k, v in self.decode_id_song.items()}
        # return [reverse_map[i] for i in recommendation_ids]
        return [
            f"{self.decode_id_artists[i][0]} - {self.decode_id_artists[i][1]}"
            for i in recommendation_ids
        ]
    # uses Levenshtein distance to make sure that we are not making
    # any spelling mistakes with song titles
    def _fuzzy_matching(self, song_artist_string):
        import re

        def clean(s):
            return re.sub(r'[^a-z0-9 ]', '', s.lower())

        song_clean = clean(song_artist_string)

        scores = []
        for idx, title_artist in self.decode_id_title_artist.items():
            ta_clean = clean(title_artist)
            ratio = fuzz.ratio(ta_clean, song_clean)
            scores.append((idx, title_artist, ratio))

        # sort descending by score
        scores.sort(key=lambda x: x[2], reverse=True)

        best_idx, best_title_artist, best_score = scores[0]

        if best_score < 40:
            print(f"Warning: weak match for '{song_artist_string}' → '{best_title_artist}' ({best_score})")
        else:
            # print(f"Good match for '{song_artist_string}' → '{best_title_artist}' ({best_score})")
            pass
        return best_idx

# Instantiate and fit the model
model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, 
                    decode_id_song=decode_id_song)

model.decode_id_artists = decode_id_artists
model.decode_id_title_artist = decode_id_title_artist

input_string = f"{SONG} - {ARTIST}"

new_recommendations = model.make_recommendation(new_song=input_string, n_recommendations=N)


### ---------------- VARUNS CODE ---------------- ###

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import requests
import http.client
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    'danceability',
    'acousticness',
    'instrumentalness',
    'loudness'
]

conn = http.client.HTTPSConnection("api.reccobeats.com")
payload = ''
headers = {
  'Accept': 'application/json'
}

def create_spotify_client():
    cred_file = "credentials.json"
    with open(cred_file, 'r') as f:
        data = json.load(f)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=data["CLIENT_ID"],
        client_secret=data["CLIENT_SECRET"]
    ))
    return sp

def get_audio_features(spotify_ids: list[str]) -> list[dict]:
    ids_param = ",".join(spotify_ids)
    url = f"https://api.reccobeats.com/v1/audio-features?ids={ids_param}"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json().get("content", [])

# returns list of user's top 20 tracks from spotify via spotipy api
def get_top_spotify_tracks() :
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

    top_tracks = sp.current_user_top_tracks(time_range='long_term', limit=20)
    return top_tracks


# takes in multiple songs info from spotipy call and gets data from reccobeats ab the song
def get_tracks_info(top_tracks):
    # extract spotify ids from top_tracks
    track_ids = [track['id'] for track in top_tracks['items']]

    # Single batch call to ReccoBeats
    response = requests.get(
        "https://api.reccobeats.com/v1/audio-features",
        params={"ids": ",".join(track_ids)},
        headers={"Accept": "application/json"}
    )
    features = response.json().get("content", [])

    print("Your Top Tracks:")
    for i, (track, feat) in enumerate(zip(top_tracks['items'], features)):
        artist_name = track['artists'][0]['name']
        track_name = track['name']
        energy = feat.get('energy')
        danceability = feat.get('danceability')
        tempo = feat.get('tempo')
        print(f"{i+1}. {artist_name} - {track_name} | energy: {energy}, danceability: {danceability}, tempo: {tempo}")
    return features


def get_track_info(track):
    track_id = track['id']
    # Single batch call to ReccoBeats
    response = requests.get(
        "https://api.reccobeats.com/v1/audio-features",
        params={"ids": ",".join(track_id)},
        headers={"Accept": "application/json"}
    )
    features = response.json().get("content", [])
    return features

"""
def top_20_with_info():
    top_tracks = get_top_spotify_tracks()
    track_features = get_tracks_info(top_tracks)
    for tf in track_features:
        print(tf)

    df = pd.DataFrame(track_features)
    print(df.info)
"""
"""
def cosine_sim():
    top_tracks = get_top_spotify_tracks()
    features = get_tracks_info(top_tracks)
    df = pd.read_csv('data/tracks_features.csv')
    print(df.head())


    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[FEATURES])

    top_df = pd.DataFrame(features)[FEATURES].fillna(0)
    top_scaled = scaler.transform(top_df)

    similarities = cosine_similarity(top_scaled, df_scaled)


    for i, track in enumerate(top_tracks['items']):
        top_indices = similarities[i].argsort()[::-1][:10]
        print(f"\nRecommendations for {track['name']}:")
        print(df.iloc[top_indices][['name', 'artists']])
"""

def get_recommendations(song_name: str, artist_name: str, df, df_scaled, scaler, n=N):
    sp = create_spotify_client()
    print("AUTH:", type(sp.auth_manager))

    match = df[
        (df['name'].str.lower() == song_name.lower()) &
        (df['artists'].str.lower().str.contains(artist_name.lower()))
        ]

    if not match.empty:
        print(f"Found '{song_name}' in local dataset")
        song_features = match[FEATURES].iloc[[0]]
        song_scaled = scaler.transform(song_features)

    else:
        print(f"Not found locally, searching Spotify...")
        sp = create_spotify_client()

        try:
            results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
        except Exception as e:
            print(f"Spotify search failed: {e}")
            return None

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

        feat = content[0]
        song_features = pd.DataFrame([feat])[FEATURES].fillna(0)
        song_scaled = scaler.transform(song_features)

    similarities = cosine_similarity(song_scaled, df_scaled)[0]
    top_indices = similarities.argsort()[::-1][:n + 20]

    print(f"\nRecommendations for '{song_name}':")
    count = 0
    for idx in top_indices:
        if idx >= len(df):
            continue
        candidate = df.iloc[idx]['name']
        artist = df.iloc[idx]['artists']
        if candidate.lower() == song_name.lower():
            continue
        print(f"{count + 1}. {artist} - {candidate}")
        count += 1
        if count == n:
            break


### ----------------- HANNAHS CODE ----------------------------- ###
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# from cosine_sim import FEATURES, get_tracks_info, get_top_spotify_tracks, get_audio_features
import spotipy
#from spotipy.oauth2 import SpotifyOAuth
import json

# TODO: make this not hard coded
K_VAL = 10
NUM_NEIGHBORS = 10

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates(subset=['id'])
    df = df.dropna()

    # extract data w/o labels
    X = df[FEATURES]
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

# def recommend_top_20():
#     scaler = StandardScaler()

#     X, df = load_data("data/tracks_features.csv")
#     scaler, km, knn, clusters, scaled_X = train_kmeans(X, scaler)

#     df['cluster'] = clusters.labels_

#     # todo: uncomment here to make actual calls to api
#     # top_tracks = get_top_spotify_tracks()
#     # track_features = get_tracks_info(top_tracks)
#     top_tracks = {'items': [{'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4C50EbCS11M0VbGyH3OfLt'}, 'href': 'https://api.spotify.com/v1/artists/4C50EbCS11M0VbGyH3OfLt', 'id': '4C50EbCS11M0VbGyH3OfLt', 'name': 'Bahamas', 'type': 'artist', 'uri': 'spotify:artist:4C50EbCS11M0VbGyH3OfLt'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/2UqlVTULRPG2qm8Bico9CK'}, 'href': 'https://api.spotify.com/v1/albums/2UqlVTULRPG2qm8Bico9CK', 'id': '2UqlVTULRPG2qm8Bico9CK', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273e1913469a3429e63f3245f9e', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02e1913469a3429e63f3245f9e', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851e1913469a3429e63f3245f9e', 'width': 64}], 'is_playable': True, 'name': 'Bahamas Is Afie', 'release_date': '2014-08-19', 'release_date_precision': 'day', 'total_tracks': 12, 'type': 'album', 'uri': 'spotify:album:2UqlVTULRPG2qm8Bico9CK'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4C50EbCS11M0VbGyH3OfLt'}, 'href': 'https://api.spotify.com/v1/artists/4C50EbCS11M0VbGyH3OfLt', 'id': '4C50EbCS11M0VbGyH3OfLt', 'name': 'Bahamas', 'type': 'artist', 'uri': 'spotify:artist:4C50EbCS11M0VbGyH3OfLt'}], 'disc_number': 1, 'duration_ms': 157040, 'explicit': False, 'external_ids': {'isrc': 'USUG11400492'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/5kgyNmIytvTGGuiv0MwzZp'}, 'href': 'https://api.spotify.com/v1/tracks/5kgyNmIytvTGGuiv0MwzZp', 'id': '5kgyNmIytvTGGuiv0MwzZp', 'is_local': False, 'is_playable': True, 'name': 'Stronger Than That', 'track_number': 5, 'type': 'track', 'uri': 'spotify:track:5kgyNmIytvTGGuiv0MwzZp'}, {'album': {'album_type': 'single', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/027TpXKGwdXP7iwbjUSpV8'}, 'href': 'https://api.spotify.com/v1/artists/027TpXKGwdXP7iwbjUSpV8', 'id': '027TpXKGwdXP7iwbjUSpV8', 'name': 'The Walters', 'type': 'artist', 'uri': 'spotify:artist:027TpXKGwdXP7iwbjUSpV8'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/7ucm85tRsWk6EyVHaYAxe9'}, 'href': 'https://api.spotify.com/v1/albums/7ucm85tRsWk6EyVHaYAxe9', 'id': '7ucm85tRsWk6EyVHaYAxe9', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273a9ab24f62c01f4bd4a08571e', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02a9ab24f62c01f4bd4a08571e', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851a9ab24f62c01f4bd4a08571e', 'width': 64}], 'is_playable': True, 'name': 'I Love You So', 'release_date': '2014-11-28', 'release_date_precision': 'day', 'total_tracks': 1, 'type': 'album', 'uri': 'spotify:album:7ucm85tRsWk6EyVHaYAxe9'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/027TpXKGwdXP7iwbjUSpV8'}, 'href': 'https://api.spotify.com/v1/artists/027TpXKGwdXP7iwbjUSpV8', 'id': '027TpXKGwdXP7iwbjUSpV8', 'name': 'The Walters', 'type': 'artist', 'uri': 'spotify:artist:027TpXKGwdXP7iwbjUSpV8'}], 'disc_number': 1, 'duration_ms': 160239, 'explicit': False, 'external_ids': {'isrc': 'TCACC1438995'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/4SqWKzw0CbA05TGszDgMlc'}, 'href': 'https://api.spotify.com/v1/tracks/4SqWKzw0CbA05TGszDgMlc', 'id': '4SqWKzw0CbA05TGszDgMlc', 'is_local': False, 'is_playable': True, 'name': 'I Love You So', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:4SqWKzw0CbA05TGszDgMlc'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6RHDASo3OVNiNY2nrGx3qc'}, 'href': 'https://api.spotify.com/v1/artists/6RHDASo3OVNiNY2nrGx3qc', 'id': '6RHDASo3OVNiNY2nrGx3qc', 'name': 'Cotton Jones', 'type': 'artist', 'uri': 'spotify:artist:6RHDASo3OVNiNY2nrGx3qc'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/44TwNftgA3QYaX3BBJAvGO'}, 'href': 'https://api.spotify.com/v1/albums/44TwNftgA3QYaX3BBJAvGO', 'id': '44TwNftgA3QYaX3BBJAvGO', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b27387db16a7855629cf7e1631d1', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e0287db16a7855629cf7e1631d1', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d0000485187db16a7855629cf7e1631d1', 'width': 64}], 'is_playable': True, 'name': 'Paranoid Cocoon', 'release_date': '2009-01-27', 'release_date_precision': 'day', 'total_tracks': 10, 'type': 'album', 'uri': 'spotify:album:44TwNftgA3QYaX3BBJAvGO'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6RHDASo3OVNiNY2nrGx3qc'}, 'href': 'https://api.spotify.com/v1/artists/6RHDASo3OVNiNY2nrGx3qc', 'id': '6RHDASo3OVNiNY2nrGx3qc', 'name': 'Cotton Jones', 'type': 'artist', 'uri': 'spotify:artist:6RHDASo3OVNiNY2nrGx3qc'}], 'disc_number': 1, 'duration_ms': 276333, 'explicit': False, 'external_ids': {'isrc': 'US33X0907908'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/2akMYW6w4sOWL1nhTzPJWu'}, 'href': 'https://api.spotify.com/v1/tracks/2akMYW6w4sOWL1nhTzPJWu', 'id': '2akMYW6w4sOWL1nhTzPJWu', 'is_local': False, 'is_playable': True, 'name': 'Blood Red Sentimental Blues', 'track_number': 9, 'type': 'track', 'uri': 'spotify:track:2akMYW6w4sOWL1nhTzPJWu'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7EGwUS3c5dXduO4sMyLWC5'}, 'href': 'https://api.spotify.com/v1/artists/7EGwUS3c5dXduO4sMyLWC5', 'id': '7EGwUS3c5dXduO4sMyLWC5', 'name': 'Houndmouth', 'type': 'artist', 'uri': 'spotify:artist:7EGwUS3c5dXduO4sMyLWC5'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/4FcLsVJY0NIKB8V9aHc1wh'}, 'href': 'https://api.spotify.com/v1/albums/4FcLsVJY0NIKB8V9aHc1wh', 'id': '4FcLsVJY0NIKB8V9aHc1wh', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273ec392389e1461e9bf3e8d0f7', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02ec392389e1461e9bf3e8d0f7', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851ec392389e1461e9bf3e8d0f7', 'width': 64}], 'is_playable': True, 'name': 'Little Neon Limelight', 'release_date': '2015-03-17', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:4FcLsVJY0NIKB8V9aHc1wh'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7EGwUS3c5dXduO4sMyLWC5'}, 'href': 'https://api.spotify.com/v1/artists/7EGwUS3c5dXduO4sMyLWC5', 'id': '7EGwUS3c5dXduO4sMyLWC5', 'name': 'Houndmouth', 'type': 'artist', 'uri': 'spotify:artist:7EGwUS3c5dXduO4sMyLWC5'}], 'disc_number': 1, 'duration_ms': 239880, 'explicit': False, 'external_ids': {'isrc': 'GBCVZ1403597'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/2MQhTbX792AT6YyzwLz9dt'}, 'href': 'https://api.spotify.com/v1/tracks/2MQhTbX792AT6YyzwLz9dt', 'id': '2MQhTbX792AT6YyzwLz9dt', 'is_local': False, 'is_playable': True, 'name': 'Sedona', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:2MQhTbX792AT6YyzwLz9dt'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7t0rwkOPGlDPEhaOcVtOt9'}, 'href': 'https://api.spotify.com/v1/artists/7t0rwkOPGlDPEhaOcVtOt9', 'id': '7t0rwkOPGlDPEhaOcVtOt9', 'name': 'The Cranberries', 'type': 'artist', 'uri': 'spotify:artist:7t0rwkOPGlDPEhaOcVtOt9'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/0AP5O47kJWlaKVnnybKvQI'}, 'href': 'https://api.spotify.com/v1/albums/0AP5O47kJWlaKVnnybKvQI', 'id': '0AP5O47kJWlaKVnnybKvQI', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273f6325f361d7803ad0d908451', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02f6325f361d7803ad0d908451', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851f6325f361d7803ad0d908451', 'width': 64}], 'is_playable': True, 'name': "Everybody Else Is Doing It, So Why Can't We?", 'release_date': '1993-03-01', 'release_date_precision': 'day', 'total_tracks': 12, 'type': 'album', 'uri': 'spotify:album:0AP5O47kJWlaKVnnybKvQI'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7t0rwkOPGlDPEhaOcVtOt9'}, 'href': 'https://api.spotify.com/v1/artists/7t0rwkOPGlDPEhaOcVtOt9', 'id': '7t0rwkOPGlDPEhaOcVtOt9', 'name': 'The Cranberries', 'type': 'artist', 'uri': 'spotify:artist:7t0rwkOPGlDPEhaOcVtOt9'}], 'disc_number': 1, 'duration_ms': 271560, 'explicit': False, 'external_ids': {'isrc': 'USIR29300080'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/4JGKZS7h4Qa16gOU3oNETV'}, 'href': 'https://api.spotify.com/v1/tracks/4JGKZS7h4Qa16gOU3oNETV', 'id': '4JGKZS7h4Qa16gOU3oNETV', 'is_local': False, 'is_playable': True, 'name': 'Dreams', 'track_number': 2, 'type': 'track', 'uri': 'spotify:track:4JGKZS7h4Qa16gOU3oNETV'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2D4FOOOtWycb3Aw9nY5n3c'}, 'href': 'https://api.spotify.com/v1/artists/2D4FOOOtWycb3Aw9nY5n3c', 'id': '2D4FOOOtWycb3Aw9nY5n3c', 'name': 'Declan McKenna', 'type': 'artist', 'uri': 'spotify:artist:2D4FOOOtWycb3Aw9nY5n3c'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/3HJiLDJgWA9Z0MvCxlzHYQ'}, 'href': 'https://api.spotify.com/v1/albums/3HJiLDJgWA9Z0MvCxlzHYQ', 'id': '3HJiLDJgWA9Z0MvCxlzHYQ', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b27311b78d26863f111044c4060f', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e0211b78d26863f111044c4060f', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d0000485111b78d26863f111044c4060f', 'width': 64}], 'is_playable': True, 'name': 'What Do You Think About the Car?', 'release_date': '2017-04-21', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:3HJiLDJgWA9Z0MvCxlzHYQ'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2D4FOOOtWycb3Aw9nY5n3c'}, 'href': 'https://api.spotify.com/v1/artists/2D4FOOOtWycb3Aw9nY5n3c', 'id': '2D4FOOOtWycb3Aw9nY5n3c', 'name': 'Declan McKenna', 'type': 'artist', 'uri': 'spotify:artist:2D4FOOOtWycb3Aw9nY5n3c'}], 'disc_number': 1, 'duration_ms': 252305, 'explicit': False, 'external_ids': {'isrc': 'GBARL1500856'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/4sNG6zQBmtq7M8aeeKJRMQ'}, 'href': 'https://api.spotify.com/v1/tracks/4sNG6zQBmtq7M8aeeKJRMQ', 'id': '4sNG6zQBmtq7M8aeeKJRMQ', 'is_local': False, 'is_playable': True, 'name': 'Brazil', 'track_number': 2, 'type': 'track', 'uri': 'spotify:track:4sNG6zQBmtq7M8aeeKJRMQ'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/3WrFJ7ztbogyGnTHbHJFl2'}, 'href': 'https://api.spotify.com/v1/artists/3WrFJ7ztbogyGnTHbHJFl2', 'id': '3WrFJ7ztbogyGnTHbHJFl2', 'name': 'The Beatles', 'type': 'artist', 'uri': 'spotify:artist:3WrFJ7ztbogyGnTHbHJFl2'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/0jTGHV5xqHPvEcwL8f6YU5'}, 'href': 'https://api.spotify.com/v1/albums/0jTGHV5xqHPvEcwL8f6YU5', 'id': '0jTGHV5xqHPvEcwL8f6YU5', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b27384243a01af3c77b56fe01ab1', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e0284243a01af3c77b56fe01ab1', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d0000485184243a01af3c77b56fe01ab1', 'width': 64}], 'is_playable': True, 'name': 'Let It Be (Remastered)', 'release_date': '1970-05-08', 'release_date_precision': 'day', 'total_tracks': 12, 'type': 'album', 'uri': 'spotify:album:0jTGHV5xqHPvEcwL8f6YU5'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/3WrFJ7ztbogyGnTHbHJFl2'}, 'href': 'https://api.spotify.com/v1/artists/3WrFJ7ztbogyGnTHbHJFl2', 'id': '3WrFJ7ztbogyGnTHbHJFl2', 'name': 'The Beatles', 'type': 'artist', 'uri': 'spotify:artist:3WrFJ7ztbogyGnTHbHJFl2'}], 'disc_number': 1, 'duration_ms': 243026, 'explicit': False, 'external_ids': {'isrc': 'GBAYE0601713'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/7iN1s7xHE4ifF5povM6A48'}, 'href': 'https://api.spotify.com/v1/tracks/7iN1s7xHE4ifF5povM6A48', 'id': '7iN1s7xHE4ifF5povM6A48', 'is_local': False, 'is_playable': True, 'name': 'Let It Be - Remastered 2009', 'track_number': 6, 'type': 'track', 'uri': 'spotify:track:7iN1s7xHE4ifF5povM6A48'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/77mJc3M7ZT5oOVM7gNdXim'}, 'href': 'https://api.spotify.com/v1/artists/77mJc3M7ZT5oOVM7gNdXim', 'id': '77mJc3M7ZT5oOVM7gNdXim', 'name': "Her's", 'type': 'artist', 'uri': 'spotify:artist:77mJc3M7ZT5oOVM7gNdXim'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/03gwRG5IvkStFnjPmgjElw'}, 'href': 'https://api.spotify.com/v1/albums/03gwRG5IvkStFnjPmgjElw', 'id': '03gwRG5IvkStFnjPmgjElw', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273fc1bc1cf80c431c2bdbde601', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02fc1bc1cf80c431c2bdbde601', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851fc1bc1cf80c431c2bdbde601', 'width': 64}], 'is_playable': True, 'name': "Songs of Her's", 'release_date': '2017-05-12', 'release_date_precision': 'day', 'total_tracks': 9, 'type': 'album', 'uri': 'spotify:album:03gwRG5IvkStFnjPmgjElw'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/77mJc3M7ZT5oOVM7gNdXim'}, 'href': 'https://api.spotify.com/v1/artists/77mJc3M7ZT5oOVM7gNdXim', 'id': '77mJc3M7ZT5oOVM7gNdXim', 'name': "Her's", 'type': 'artist', 'uri': 'spotify:artist:77mJc3M7ZT5oOVM7gNdXim'}], 'disc_number': 1, 'duration_ms': 255067, 'explicit': False, 'external_ids': {'isrc': 'GBYEJ1100223'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/1XrSjpNe49IiygZfzb74pk'}, 'href': 'https://api.spotify.com/v1/tracks/1XrSjpNe49IiygZfzb74pk', 'id': '1XrSjpNe49IiygZfzb74pk', 'is_local': False, 'is_playable': True, 'name': 'What Once Was', 'track_number': 7, 'type': 'track', 'uri': 'spotify:track:1XrSjpNe49IiygZfzb74pk'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4mLJ3XfOM5FPjSAWdQ2Jk7'}, 'href': 'https://api.spotify.com/v1/artists/4mLJ3XfOM5FPjSAWdQ2Jk7', 'id': '4mLJ3XfOM5FPjSAWdQ2Jk7', 'name': 'Dr. Dog', 'type': 'artist', 'uri': 'spotify:artist:4mLJ3XfOM5FPjSAWdQ2Jk7'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/16XswZ18xhMs8qUTN51mRl'}, 'href': 'https://api.spotify.com/v1/albums/16XswZ18xhMs8qUTN51mRl', 'id': '16XswZ18xhMs8qUTN51mRl', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2730478062bc04df0947d232fcb', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e020478062bc04df0947d232fcb', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048510478062bc04df0947d232fcb', 'width': 64}], 'is_playable': True, 'name': 'Shame, Shame (Deluxe Edition)', 'release_date': '2010-11-02', 'release_date_precision': 'day', 'total_tracks': 18, 'type': 'album', 'uri': 'spotify:album:16XswZ18xhMs8qUTN51mRl'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4mLJ3XfOM5FPjSAWdQ2Jk7'}, 'href': 'https://api.spotify.com/v1/artists/4mLJ3XfOM5FPjSAWdQ2Jk7', 'id': '4mLJ3XfOM5FPjSAWdQ2Jk7', 'name': 'Dr. Dog', 'type': 'artist', 'uri': 'spotify:artist:4mLJ3XfOM5FPjSAWdQ2Jk7'}], 'disc_number': 1, 'duration_ms': 234800, 'explicit': False, 'external_ids': {'isrc': 'USEP40937005'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/0UV5zxRMz6AO4ZwUOZNIKI'}, 'href': 'https://api.spotify.com/v1/tracks/0UV5zxRMz6AO4ZwUOZNIKI', 'id': '0UV5zxRMz6AO4ZwUOZNIKI', 'is_local': False, 'is_playable': True, 'name': "Where'd All the Time Go?", 'track_number': 5, 'type': 'track', 'uri': 'spotify:track:0UV5zxRMz6AO4ZwUOZNIKI'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6Qm9stX6XO1a4c7BXQDDgc'}, 'href': 'https://api.spotify.com/v1/artists/6Qm9stX6XO1a4c7BXQDDgc', 'id': '6Qm9stX6XO1a4c7BXQDDgc', 'name': 'Fruit Bats', 'type': 'artist', 'uri': 'spotify:artist:6Qm9stX6XO1a4c7BXQDDgc'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/4fu8RdgNHUGQ61GP0sILpp'}, 'href': 'https://api.spotify.com/v1/albums/4fu8RdgNHUGQ61GP0sILpp', 'id': '4fu8RdgNHUGQ61GP0sILpp', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273a4b2fcab8ba091bf1003d6b1', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02a4b2fcab8ba091bf1003d6b1', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851a4b2fcab8ba091bf1003d6b1', 'width': 64}], 'is_playable': True, 'name': 'Gold Past Life', 'release_date': '2019-06-21', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:4fu8RdgNHUGQ61GP0sILpp'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6Qm9stX6XO1a4c7BXQDDgc'}, 'href': 'https://api.spotify.com/v1/artists/6Qm9stX6XO1a4c7BXQDDgc', 'id': '6Qm9stX6XO1a4c7BXQDDgc', 'name': 'Fruit Bats', 'type': 'artist', 'uri': 'spotify:artist:6Qm9stX6XO1a4c7BXQDDgc'}], 'disc_number': 1, 'duration_ms': 181400, 'explicit': False, 'external_ids': {'isrc': 'USMRG1967001'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/6tZetCGfhxPh5ZIKCGmaKq'}, 'href': 'https://api.spotify.com/v1/tracks/6tZetCGfhxPh5ZIKCGmaKq', 'id': '6tZetCGfhxPh5ZIKCGmaKq', 'is_local': False, 'is_playable': True, 'name': 'The Bottom of It', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:6tZetCGfhxPh5ZIKCGmaKq'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/0LcJLqbBmaGUft1e9Mm8HV'}, 'href': 'https://api.spotify.com/v1/artists/0LcJLqbBmaGUft1e9Mm8HV', 'id': '0LcJLqbBmaGUft1e9Mm8HV', 'name': 'ABBA', 'type': 'artist', 'uri': 'spotify:artist:0LcJLqbBmaGUft1e9Mm8HV'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/5GByGaws7Cw0t28kjvAOzV'}, 'href': 'https://api.spotify.com/v1/albums/5GByGaws7Cw0t28kjvAOzV', 'id': '5GByGaws7Cw0t28kjvAOzV', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b27342678f54dfd1d5afb3eea19a', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e0242678f54dfd1d5afb3eea19a', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d0000485142678f54dfd1d5afb3eea19a', 'width': 64}], 'is_playable': True, 'name': 'The Visitors (Deluxe Edition)', 'release_date': '1981', 'release_date_precision': 'year', 'total_tracks': 16, 'type': 'album', 'uri': 'spotify:album:5GByGaws7Cw0t28kjvAOzV'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/0LcJLqbBmaGUft1e9Mm8HV'}, 'href': 'https://api.spotify.com/v1/artists/0LcJLqbBmaGUft1e9Mm8HV', 'id': '0LcJLqbBmaGUft1e9Mm8HV', 'name': 'ABBA', 'type': 'artist', 'uri': 'spotify:artist:0LcJLqbBmaGUft1e9Mm8HV'}], 'disc_number': 1, 'duration_ms': 233720, 'explicit': False, 'external_ids': {'isrc': 'SEAYD8102080'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/6TvxPS4fj4LUdjw2es4g21'}, 'href': 'https://api.spotify.com/v1/tracks/6TvxPS4fj4LUdjw2es4g21', 'id': '6TvxPS4fj4LUdjw2es4g21', 'is_local': False, 'is_playable': True, 'name': 'Slipping Through My Fingers', 'track_number': 8, 'type': 'track', 'uri': 'spotify:track:6TvxPS4fj4LUdjw2es4g21'}, {'album': {'album_type': 'single', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6FugQjLquBF4JzATRN70bR'}, 'href': 'https://api.spotify.com/v1/artists/6FugQjLquBF4JzATRN70bR', 'id': '6FugQjLquBF4JzATRN70bR', 'name': 'Yot Club', 'type': 'artist', 'uri': 'spotify:artist:6FugQjLquBF4JzATRN70bR'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/5AyeXsonrznZMqynW61gY9'}, 'href': 'https://api.spotify.com/v1/albums/5AyeXsonrznZMqynW61gY9', 'id': '5AyeXsonrznZMqynW61gY9', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2732f9462a6732046829db5683d', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e022f9462a6732046829db5683d', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048512f9462a6732046829db5683d', 'width': 64}], 'is_playable': True, 'name': 'Fly Out West', 'release_date': '2019-05-31', 'release_date_precision': 'day', 'total_tracks': 1, 'type': 'album', 'uri': 'spotify:album:5AyeXsonrznZMqynW61gY9'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6FugQjLquBF4JzATRN70bR'}, 'href': 'https://api.spotify.com/v1/artists/6FugQjLquBF4JzATRN70bR', 'id': '6FugQjLquBF4JzATRN70bR', 'name': 'Yot Club', 'type': 'artist', 'uri': 'spotify:artist:6FugQjLquBF4JzATRN70bR'}], 'disc_number': 1, 'duration_ms': 174211, 'explicit': False, 'external_ids': {'isrc': 'CAGOO1906218'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/0ASIqnVJvN1GmH1xEBdf2a'}, 'href': 'https://api.spotify.com/v1/tracks/0ASIqnVJvN1GmH1xEBdf2a', 'id': '0ASIqnVJvN1GmH1xEBdf2a', 'is_local': False, 'is_playable': True, 'name': 'Fly Out West - Single Version', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:0ASIqnVJvN1GmH1xEBdf2a'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/251UrhgNbMr15NLzQ2KyKq'}, 'href': 'https://api.spotify.com/v1/artists/251UrhgNbMr15NLzQ2KyKq', 'id': '251UrhgNbMr15NLzQ2KyKq', 'name': 'Rayland Baxter', 'type': 'artist', 'uri': 'spotify:artist:251UrhgNbMr15NLzQ2KyKq'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/72YUTJrAuTuSHSVrgixbor'}, 'href': 'https://api.spotify.com/v1/albums/72YUTJrAuTuSHSVrgixbor', 'id': '72YUTJrAuTuSHSVrgixbor', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2733e896ecce01a0b06ee0d8576', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e023e896ecce01a0b06ee0d8576', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048513e896ecce01a0b06ee0d8576', 'width': 64}], 'is_playable': True, 'name': 'Imaginary Man', 'release_date': '2015-08-14', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:72YUTJrAuTuSHSVrgixbor'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/251UrhgNbMr15NLzQ2KyKq'}, 'href': 'https://api.spotify.com/v1/artists/251UrhgNbMr15NLzQ2KyKq', 'id': '251UrhgNbMr15NLzQ2KyKq', 'name': 'Rayland Baxter', 'type': 'artist', 'uri': 'spotify:artist:251UrhgNbMr15NLzQ2KyKq'}], 'disc_number': 1, 'duration_ms': 230786, 'explicit': False, 'external_ids': {'isrc': 'USATO1400776'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/3HOXNIj8NjlgjQiBd3YVIi'}, 'href': 'https://api.spotify.com/v1/tracks/3HOXNIj8NjlgjQiBd3YVIi', 'id': '3HOXNIj8NjlgjQiBd3YVIi', 'is_local': False, 'is_playable': True, 'name': 'Yellow Eyes', 'track_number': 4, 'type': 'track', 'uri': 'spotify:track:3HOXNIj8NjlgjQiBd3YVIi'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6fC2AcsQtd9h4BWELbbire'}, 'href': 'https://api.spotify.com/v1/artists/6fC2AcsQtd9h4BWELbbire', 'id': '6fC2AcsQtd9h4BWELbbire', 'name': 'Peach Pit', 'type': 'artist', 'uri': 'spotify:artist:6fC2AcsQtd9h4BWELbbire'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/291A3Ud0sbMSfmG48k6GQY'}, 'href': 'https://api.spotify.com/v1/albums/291A3Ud0sbMSfmG48k6GQY', 'id': '291A3Ud0sbMSfmG48k6GQY', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273e271020eb2d3fa240f0f51c6', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02e271020eb2d3fa240f0f51c6', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851e271020eb2d3fa240f0f51c6', 'width': 64}], 'is_playable': True, 'name': 'From 2 to 3', 'release_date': '2022-03-04', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:291A3Ud0sbMSfmG48k6GQY'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6fC2AcsQtd9h4BWELbbire'}, 'href': 'https://api.spotify.com/v1/artists/6fC2AcsQtd9h4BWELbbire', 'id': '6fC2AcsQtd9h4BWELbbire', 'name': 'Peach Pit', 'type': 'artist', 'uri': 'spotify:artist:6fC2AcsQtd9h4BWELbbire'}], 'disc_number': 1, 'duration_ms': 230747, 'explicit': False, 'external_ids': {'isrc': 'USSM12102947'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/1qbjCGGsZpEq56QgvtJoX7'}, 'href': 'https://api.spotify.com/v1/tracks/1qbjCGGsZpEq56QgvtJoX7', 'id': '1qbjCGGsZpEq56QgvtJoX7', 'is_local': False, 'is_playable': True, 'name': 'Up Granville', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:1qbjCGGsZpEq56QgvtJoX7'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4LEiUm1SRbFMgfqnQTwUbQ'}, 'href': 'https://api.spotify.com/v1/artists/4LEiUm1SRbFMgfqnQTwUbQ', 'id': '4LEiUm1SRbFMgfqnQTwUbQ', 'name': 'Bon Iver', 'type': 'artist', 'uri': 'spotify:artist:4LEiUm1SRbFMgfqnQTwUbQ'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/7EJ0OT5ZqybXxcYRa6mccM'}, 'href': 'https://api.spotify.com/v1/albums/7EJ0OT5ZqybXxcYRa6mccM', 'id': '7EJ0OT5ZqybXxcYRa6mccM', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273bf7c317a63c4f128b8823406', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02bf7c317a63c4f128b8823406', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851bf7c317a63c4f128b8823406', 'width': 64}], 'is_playable': True, 'name': 'For Emma, Forever Ago', 'release_date': '2008-02-19', 'release_date_precision': 'day', 'total_tracks': 9, 'type': 'album', 'uri': 'spotify:album:7EJ0OT5ZqybXxcYRa6mccM'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4LEiUm1SRbFMgfqnQTwUbQ'}, 'href': 'https://api.spotify.com/v1/artists/4LEiUm1SRbFMgfqnQTwUbQ', 'id': '4LEiUm1SRbFMgfqnQTwUbQ', 'name': 'Bon Iver', 'type': 'artist', 'uri': 'spotify:artist:4LEiUm1SRbFMgfqnQTwUbQ'}], 'disc_number': 1, 'duration_ms': 220720, 'explicit': False, 'external_ids': {'isrc': 'US38Y0811508'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/2nilAlGEZmwyaLTMMyDdLo'}, 'href': 'https://api.spotify.com/v1/tracks/2nilAlGEZmwyaLTMMyDdLo', 'id': '2nilAlGEZmwyaLTMMyDdLo', 'is_local': False, 'is_playable': True, 'name': 'For Emma', 'track_number': 8, 'type': 'track', 'uri': 'spotify:track:2nilAlGEZmwyaLTMMyDdLo'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/48vDIufGC8ujPuBiTxY8dm'}, 'href': 'https://api.spotify.com/v1/artists/48vDIufGC8ujPuBiTxY8dm', 'id': '48vDIufGC8ujPuBiTxY8dm', 'name': 'Palace', 'type': 'artist', 'uri': 'spotify:artist:48vDIufGC8ujPuBiTxY8dm'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/6cmFNl8lllA6BGc7SKLy3y'}, 'href': 'https://api.spotify.com/v1/albums/6cmFNl8lllA6BGc7SKLy3y', 'id': '6cmFNl8lllA6BGc7SKLy3y', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273929dae46c6b93942c7499b7d', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02929dae46c6b93942c7499b7d', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851929dae46c6b93942c7499b7d', 'width': 64}], 'is_playable': True, 'name': 'So Long Forever', 'release_date': '2016-11-04', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:6cmFNl8lllA6BGc7SKLy3y'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/48vDIufGC8ujPuBiTxY8dm'}, 'href': 'https://api.spotify.com/v1/artists/48vDIufGC8ujPuBiTxY8dm', 'id': '48vDIufGC8ujPuBiTxY8dm', 'name': 'Palace', 'type': 'artist', 'uri': 'spotify:artist:48vDIufGC8ujPuBiTxY8dm'}], 'disc_number': 1, 'duration_ms': 249840, 'explicit': False, 'external_ids': {'isrc': 'GBUM71603029'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/2H30WL3exSctlDC9GyRbD4'}, 'href': 'https://api.spotify.com/v1/tracks/2H30WL3exSctlDC9GyRbD4', 'id': '2H30WL3exSctlDC9GyRbD4', 'is_local': False, 'is_playable': True, 'name': 'Live Well', 'track_number': 3, 'type': 'track', 'uri': 'spotify:track:2H30WL3exSctlDC9GyRbD4'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4EVpmkEwrLYEg6jIsiPMIb'}, 'href': 'https://api.spotify.com/v1/artists/4EVpmkEwrLYEg6jIsiPMIb', 'id': '4EVpmkEwrLYEg6jIsiPMIb', 'name': 'Fleet Foxes', 'type': 'artist', 'uri': 'spotify:artist:4EVpmkEwrLYEg6jIsiPMIb'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/5GRnydamKvIeG46dycID6v'}, 'href': 'https://api.spotify.com/v1/albums/5GRnydamKvIeG46dycID6v', 'id': '5GRnydamKvIeG46dycID6v', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2733818b4c636e2a7fdea3bf965', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e023818b4c636e2a7fdea3bf965', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048513818b4c636e2a7fdea3bf965', 'width': 64}], 'is_playable': True, 'name': 'Fleet Foxes', 'release_date': '2008-06-03', 'release_date_precision': 'day', 'total_tracks': 11, 'type': 'album', 'uri': 'spotify:album:5GRnydamKvIeG46dycID6v'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4EVpmkEwrLYEg6jIsiPMIb'}, 'href': 'https://api.spotify.com/v1/artists/4EVpmkEwrLYEg6jIsiPMIb', 'id': '4EVpmkEwrLYEg6jIsiPMIb', 'name': 'Fleet Foxes', 'type': 'artist', 'uri': 'spotify:artist:4EVpmkEwrLYEg6jIsiPMIb'}], 'disc_number': 1, 'duration_ms': 147026, 'explicit': False, 'external_ids': {'isrc': 'USSUB0877702'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/0GegHVxeozw3rdjte45Bfx'}, 'href': 'https://api.spotify.com/v1/tracks/0GegHVxeozw3rdjte45Bfx', 'id': '0GegHVxeozw3rdjte45Bfx', 'is_local': False, 'is_playable': True, 'name': 'White Winter Hymnal', 'track_number': 2, 'type': 'track', 'uri': 'spotify:track:0GegHVxeozw3rdjte45Bfx'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/69tiO1fG8VWduDl3ji2qhI'}, 'href': 'https://api.spotify.com/v1/artists/69tiO1fG8VWduDl3ji2qhI', 'id': '69tiO1fG8VWduDl3ji2qhI', 'name': 'Mt. Joy', 'type': 'artist', 'uri': 'spotify:artist:69tiO1fG8VWduDl3ji2qhI'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/4epZkfflxt4dFZpgafIx01'}, 'href': 'https://api.spotify.com/v1/albums/4epZkfflxt4dFZpgafIx01', 'id': '4epZkfflxt4dFZpgafIx01', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273a877e4caf6e989c666799fb9', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02a877e4caf6e989c666799fb9', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851a877e4caf6e989c666799fb9', 'width': 64}], 'is_playable': True, 'name': 'Rearrange Us', 'release_date': '2020-06-05', 'release_date_precision': 'day', 'total_tracks': 13, 'type': 'album', 'uri': 'spotify:album:4epZkfflxt4dFZpgafIx01'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/69tiO1fG8VWduDl3ji2qhI'}, 'href': 'https://api.spotify.com/v1/artists/69tiO1fG8VWduDl3ji2qhI', 'id': '69tiO1fG8VWduDl3ji2qhI', 'name': 'Mt. Joy', 'type': 'artist', 'uri': 'spotify:artist:69tiO1fG8VWduDl3ji2qhI'}], 'disc_number': 1, 'duration_ms': 193186, 'explicit': False, 'external_ids': {'isrc': 'USDMG2000613'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/1bGJz16oDQgDOeTFaLm8Nz'}, 'href': 'https://api.spotify.com/v1/tracks/1bGJz16oDQgDOeTFaLm8Nz', 'id': '1bGJz16oDQgDOeTFaLm8Nz', 'is_local': False, 'is_playable': True, 'name': 'Strangers', 'track_number': 13, 'type': 'track', 'uri': 'spotify:track:1bGJz16oDQgDOeTFaLm8Nz'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6Pled8lBAODAviS574l1Q4'}, 'href': 'https://api.spotify.com/v1/artists/6Pled8lBAODAviS574l1Q4', 'id': '6Pled8lBAODAviS574l1Q4', 'name': 'Night Moves', 'type': 'artist', 'uri': 'spotify:artist:6Pled8lBAODAviS574l1Q4'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/4QH2Ppf0BHxK8mGVF6aEmD'}, 'href': 'https://api.spotify.com/v1/albums/4QH2Ppf0BHxK8mGVF6aEmD', 'id': '4QH2Ppf0BHxK8mGVF6aEmD', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273fd0ea87c616bf1db825bf66d', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02fd0ea87c616bf1db825bf66d', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851fd0ea87c616bf1db825bf66d', 'width': 64}], 'is_playable': True, 'name': 'Colored Emotions', 'release_date': '2012-10-16', 'release_date_precision': 'day', 'total_tracks': 10, 'type': 'album', 'uri': 'spotify:album:4QH2Ppf0BHxK8mGVF6aEmD'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6Pled8lBAODAviS574l1Q4'}, 'href': 'https://api.spotify.com/v1/artists/6Pled8lBAODAviS574l1Q4', 'id': '6Pled8lBAODAviS574l1Q4', 'name': 'Night Moves', 'type': 'artist', 'uri': 'spotify:artist:6Pled8lBAODAviS574l1Q4'}], 'disc_number': 1, 'duration_ms': 259866, 'explicit': False, 'external_ids': {'isrc': 'GBCEL1200233'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/1YCt0tIYzY5zNWAF9HkaqG'}, 'href': 'https://api.spotify.com/v1/tracks/1YCt0tIYzY5zNWAF9HkaqG', 'id': '1YCt0tIYzY5zNWAF9HkaqG', 'is_local': False, 'is_playable': True, 'name': 'Colored Emotions', 'track_number': 10, 'type': 'track', 'uri': 'spotify:track:1YCt0tIYzY5zNWAF9HkaqG'}, {'album': {'album_type': 'album', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6eJa3zG1QZLRB3xgRuyxbm'}, 'href': 'https://api.spotify.com/v1/artists/6eJa3zG1QZLRB3xgRuyxbm', 'id': '6eJa3zG1QZLRB3xgRuyxbm', 'name': 'Dayglow', 'type': 'artist', 'uri': 'spotify:artist:6eJa3zG1QZLRB3xgRuyxbm'}], 'external_urls': {'spotify': 'https://open.spotify.com/album/7GYzQIMfdDWo2XC4BDLHPk'}, 'href': 'https://api.spotify.com/v1/albums/7GYzQIMfdDWo2XC4BDLHPk', 'id': '7GYzQIMfdDWo2XC4BDLHPk', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2735160eaecb31b739ea1c2eaa5', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e025160eaecb31b739ea1c2eaa5', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048515160eaecb31b739ea1c2eaa5', 'width': 64}], 'is_playable': True, 'name': 'Fuzzybrain', 'release_date': '2019-11-14', 'release_date_precision': 'day', 'total_tracks': 10, 'type': 'album', 'uri': 'spotify:album:7GYzQIMfdDWo2XC4BDLHPk'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/6eJa3zG1QZLRB3xgRuyxbm'}, 'href': 'https://api.spotify.com/v1/artists/6eJa3zG1QZLRB3xgRuyxbm', 'id': '6eJa3zG1QZLRB3xgRuyxbm', 'name': 'Dayglow', 'type': 'artist', 'uri': 'spotify:artist:6eJa3zG1QZLRB3xgRuyxbm'}], 'disc_number': 1, 'duration_ms': 252400, 'explicit': False, 'external_ids': {'isrc': 'TCADU1861917'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/08TjqEEAO32VuF002ePbTz'}, 'href': 'https://api.spotify.com/v1/tracks/08TjqEEAO32VuF002ePbTz', 'id': '08TjqEEAO32VuF002ePbTz', 'is_local': False, 'is_playable': True, 'name': 'Junior Varsity', 'track_number': 8, 'type': 'track', 'uri': 'spotify:track:08TjqEEAO32VuF002ePbTz'}], 'total': 8171, 'limit': 20, 'offset': 0, 'href': 'https://api.spotify.com/v1/me/top/tracks?offset=0&limit=20&time_range=long_term', 'next': 'https://api.spotify.com/v1/me/top/tracks?offset=20&limit=20&time_range=long_term', 'previous': None}
#     track_features = [
#         {'id': '7e60e505-4f53-45e8-8871-078a3c9a05b2', 'href': 'https://open.spotify.com/track/1bGJz16oDQgDOeTFaLm8Nz', 'isrc': 'USDMG2000613', 'acousticness': 0.739, 'danceability': 0.57, 'energy': 0.636, 'instrumentalness': 0.0221, 'key': 7, 'liveness': 0.0807, 'loudness': -8.136, 'mode': 1, 'speechiness': 0.0396, 'tempo': 101.984, 'valence': 0.219},
#         {'id': 'aade5391-0ae0-47d8-9681-63e9238c6407', 'href': 'https://open.spotify.com/track/4SqWKzw0CbA05TGszDgMlc', 'isrc': 'TCACC1438995', 'acousticness': 0.583, 'danceability': 0.575, 'energy': 0.648, 'instrumentalness': 0.0, 'key': 10, 'liveness': 0.115, 'loudness': -4.891, 'mode': 1, 'speechiness': 0.0358, 'tempo': 75.977, 'valence': 0.466},
#         {'id': '70ee017e-3936-4951-ad39-9e32d49edfe9', 'href': 'https://open.spotify.com/track/08TjqEEAO32VuF002ePbTz', 'isrc': 'TCADU1861917', 'acousticness': 0.0835, 'danceability': 0.259, 'energy': 0.521, 'instrumentalness': 0.0474, 'key': 8, 'liveness': 0.089, 'loudness': -8.022, 'mode': 1, 'speechiness': 0.0489, 'tempo': 75.003, 'valence': 0.267},
#         {'id': 'f19059af-5569-4453-b455-f17db97d1d87', 'href': 'https://open.spotify.com/track/0UV5zxRMz6AO4ZwUOZNIKI', 'isrc': 'USEP40937005', 'acousticness': 0.132, 'danceability': 0.454, 'energy': 0.82, 'instrumentalness': 0.000969, 'key': 2, 'liveness': 0.115, 'loudness': -4.193, 'mode': 1, 'speechiness': 0.0567, 'tempo': 166.303, 'valence': 0.575},
#         {'id': '5b54e267-7b83-4fa1-9729-303779ba25e5', 'href': 'https://open.spotify.com/track/7iN1s7xHE4ifF5povM6A48', 'isrc': 'GBAYE0601713', 'acousticness': 0.631, 'danceability': 0.443, 'energy': 0.403, 'instrumentalness': 0.0, 'key': 0, 'liveness': 0.111, 'loudness': -8.339, 'mode': 1, 'speechiness': 0.0322, 'tempo': 143.462, 'valence': 0.41},
#         {'id': 'caf171d8-543b-4c6c-9402-a5850354897c', 'href': 'https://open.spotify.com/track/6TvxPS4fj4LUdjw2es4g21', 'isrc': 'SEAYD8102080', 'acousticness': 0.796, 'danceability': 0.475, 'energy': 0.26, 'instrumentalness': 0.0016, 'key': 5, 'liveness': 0.109, 'loudness': -15.997, 'mode': 1, 'speechiness': 0.0322, 'tempo': 137.212, 'valence': 0.339},
#         {'id': '9728a30e-4184-4693-91cc-8642cd2e1155', 'href': 'https://open.spotify.com/track/4JGKZS7h4Qa16gOU3oNETV', 'isrc': 'USIR29300080', 'acousticness': 0.0031, 'danceability': 0.551, 'energy': 0.645, 'instrumentalness': 0.00376, 'key': 4, 'liveness': 0.421, 'loudness': -13.093, 'mode': 1, 'speechiness': 0.0354, 'tempo': 128.665, 'valence': 0.508},
#         {'id': '85aa6d37-3f77-46b8-ace1-406b1ccd16c3', 'href': 'https://open.spotify.com/track/2nilAlGEZmwyaLTMMyDdLo', 'isrc': 'US38Y0811508', 'acousticness': 0.487, 'danceability': 0.669, 'energy': 0.613, 'instrumentalness': 0.176, 'key': 4, 'liveness': 0.132, 'loudness': -11.12, 'mode': 0, 'speechiness': 0.036, 'tempo': 110.668, 'valence': 0.566},
#         {'id': '3266c023-1e0d-4a6e-a0d1-ec4082fbc399', 'href': 'https://open.spotify.com/track/0ASIqnVJvN1GmH1xEBdf2a', 'isrc': 'CAGOO1906218', 'acousticness': 0.417, 'danceability': 0.499, 'energy': 0.666, 'instrumentalness': 0.461, 'key': 2, 'liveness': 0.0715, 'loudness': -15.61, 'mode': 1, 'speechiness': 0.0446, 'tempo': 173.325, 'valence': 0.405},
#         {'id': 'd6a2693e-ad37-4027-b64d-88ea815f0f5d', 'href': 'https://open.spotify.com/track/0GegHVxeozw3rdjte45Bfx', 'isrc': 'USSUB0877702', 'acousticness': 0.44, 'danceability': 0.628, 'energy': 0.5, 'instrumentalness': 0.0, 'key': 6, 'liveness': 0.244, 'loudness': -9.66, 'mode': 0, 'speechiness': 0.0268, 'tempo': 124.932, 'valence': 0.68},
#         {'id': '2c95dc0e-031b-47f9-a758-87aeb2127534', 'href': 'https://open.spotify.com/track/2akMYW6w4sOWL1nhTzPJWu', 'isrc': 'US33X0907908', 'acousticness': 0.218, 'danceability': 0.565, 'energy': 0.716, 'instrumentalness': 3.3e-06, 'key': 10, 'liveness': 0.185, 'loudness': -6.254, 'mode': 1, 'speechiness': 0.0253, 'tempo': 92.977, 'valence': 0.679},
#         {'id': 'edc3b775-0319-4ba0-a4e2-e64fc655e37f', 'href': 'https://open.spotify.com/track/3HOXNIj8NjlgjQiBd3YVIi', 'isrc': 'USATO1400776', 'acousticness': 0.0948, 'danceability': 0.496, 'energy': 0.679, 'instrumentalness': 0.0, 'key': 10, 'liveness': 0.103, 'loudness': -7.898, 'mode': 1, 'speechiness': 0.0368, 'tempo': 154.028, 'valence': 0.507},
#         {'id': 'cf821d42-6bc3-470e-a2e9-4ced929b0be8', 'href': 'https://open.spotify.com/track/2H30WL3exSctlDC9GyRbD4', 'isrc': 'GBUM71603029', 'acousticness': 0.264, 'danceability': 0.496, 'energy': 0.644, 'instrumentalness': 0.0, 'key': 2, 'liveness': 0.0695, 'loudness': -5.385, 'mode': 1, 'speechiness': 0.0269, 'tempo': 96.017, 'valence': 0.435},
#         {'id': 'df747051-32c7-448e-a2a7-f91cbe12ad09', 'href': 'https://open.spotify.com/track/6tZetCGfhxPh5ZIKCGmaKq', 'isrc': 'USMRG1967001', 'acousticness': 0.134, 'danceability': 0.631, 'energy': 0.625, 'instrumentalness': 0.0196, 'key': 0, 'liveness': 0.117, 'loudness': -10.73, 'mode': 1, 'speechiness': 0.0374, 'tempo': 155.987, 'valence': 0.305},
#         {'id': 'dcff22e3-14ea-489a-bbca-e449c5245e9d', 'href': 'https://open.spotify.com/track/4sNG6zQBmtq7M8aeeKJRMQ', 'isrc': 'GBARL1500856', 'acousticness': 0.0941, 'danceability': 0.687, 'energy': 0.617, 'instrumentalness': 1.27e-05, 'key': 4, 'liveness': 0.0898, 'loudness': -5.213, 'mode': 1, 'speechiness': 0.0287, 'tempo': 121.079, 'valence': 0.665},
#         {'id': '080149fa-533a-4e4a-b59e-cea6428f9d55', 'href': 'https://open.spotify.com/track/1qbjCGGsZpEq56QgvtJoX7', 'isrc': 'USSM12102947', 'acousticness': 0.171, 'danceability': 0.646, 'energy': 0.611, 'instrumentalness': 0.0, 'key': 0, 'liveness': 0.0634, 'loudness': -6.516, 'mode': 1, 'speechiness': 0.0272, 'tempo': 109.051, 'valence': 0.272},
#         {'id': 'd9beca41-edb0-4e47-86bc-a8ffd31c6700', 'href': 'https://open.spotify.com/track/5kgyNmIytvTGGuiv0MwzZp', 'isrc': 'USUG11400492', 'acousticness': 0.432, 'danceability': 0.745, 'energy': 0.539, 'instrumentalness': 0.00162, 'key': 0, 'liveness': 0.0826, 'loudness': -4.41, 'mode': 1, 'speechiness': 0.0271, 'tempo': 95.893, 'valence': 0.886}
#     ]

#     # scale song features from spotify/reccobeats calls
#     track_features_df = pd.DataFrame(track_features)[FEATURES]

#     scaled_feats = scaler.transform(track_features_df.values)
#     distances, idxs = knn.kneighbors(scaled_feats)

#     # TODO: make sure that there are no duplicate songs in the output
#     # if the number of songs in spotify's top tracks != number of songs you are recommended for
#     if len(top_tracks['items']) > len(idxs):
#         top_tracks['items'] = top_tracks['items'][:len(idxs)]

#     for i, track in enumerate(top_tracks['items']):
#         print(f"\nRecommendations for {track['name']} by {track['artists'][0]['name']}:")
#         format_recs(df.iloc[idxs[i]])

def input_to_rec(song_info):
    scaler = StandardScaler()

    X, df = load_data("data/tracks_features.csv")
    scaler, km, knn, clusters, scaled_X = train_kmeans(X, scaler)

    df['cluster'] = clusters.labels_

    if isinstance(song_info, pd.Series):
        track_features_df = song_info[FEATURES].to_frame().T

    elif isinstance(song_info, dict):
        track_features_df = pd.DataFrame([song_info])[FEATURES]

    elif isinstance(song_info, pd.DataFrame):
        track_features_df = song_info[FEATURES].iloc[[0]]

    elif isinstance(song_info, list):
        if len(song_info) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} feature values, got {len(song_info)}")
        track_features_df = pd.DataFrame([song_info], columns=FEATURES)

    else:
        raise TypeError(f"song_info has unexpected type: {type(song_info)}")

    scaled_feats = scaler.transform(track_features_df)
    distances, idxs = knn.kneighbors(scaled_feats)

    return df.iloc[idxs[0]]

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

    return f"\t{title_str} by {artists_str}"



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


def get_one_song_feats(song_name, artist_name, sp):
    try:
        results = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
    except Exception as e:
        print(f"Spotify search failed in get_one_song_feats: {e}")
        return None

    if not results['tracks']['items']:
        return None

    return results['tracks']['items'][0]





    # song = "Untouched"
    # artist = "The Veronicas"
    # sp_id = get_one_song_feats(song, artist)
    # song_with_feats = get_audio_features([sp_id['id']])
    # print(song_with_feats)

    # done = False

    # while not done:
    #     song = input("Enter song name: ")
    #     artist = input("Enter artist name: ")
    #     sp_id = get_one_song_feats(song, artist)
    #     song_with_feats = get_audio_features([sp_id['id']])
    #     print(song_with_feats)

    #     all_recs = input_to_rec(song_with_feats)
    #     # return the first one
    #     print(all_recs[0])
    #     print(f"\nRecommendation for {song} by {artist}: {all_recs[0]}")

    #     cont = input("Continue? (y/n): ")
    #     if cont != "y":
    #         done = True









# FINAL OUTPUT
if __name__ == "__main__":
    df = pd.read_csv('data/tracks_features.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[FEATURES])

    print(f"Recommendations for {SONG} by {ARTIST}:")
    get_recommendations(SONG, ARTIST, df, df_scaled, scaler)
    print("\n".join(new_recommendations))

    done = False
    while not done:
        sp = create_spotify_client()

        sp_id = get_one_song_feats(SONG, ARTIST, sp)
        if sp_id is None:
            print(f"Could not fetch song info for {SONG} by {ARTIST}.")
            break

        feats_list = get_audio_features([sp_id['id']])
        if not feats_list:
            print(f"No ReccoBeats features found for {SONG} by {ARTIST}.")
            break

        song_with_feats = feats_list[0]
        all_recs = input_to_rec(song_with_feats)

        formatted = format_recs(all_recs)
        print(f"\nRecommendation for {SONG} by {ARTIST}: {formatted[0]}")

        cont = input("Continue? (y/n): ")
        if cont != "y":
            done = True