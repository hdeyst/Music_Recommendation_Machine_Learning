import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from api_song_loading import FEATURES, get_tracks_info, get_top_spotify_tracks
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

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

    df['cluster'] = clusters.labels_
    print("dataframe and info")
    print(df)
    print(df.columns)

    # top_tracks = get_top_spotify_tracks()
    # track_features = get_tracks_info(top_tracks)
    # todo: here so don't call api each time
    track_features = [{'id': '7e60e505-4f53-45e8-8871-078a3c9a05b2', 'href': 'https://open.spotify.com/track/1bGJz16oDQgDOeTFaLm8Nz', 'isrc': 'USDMG2000613', 'acousticness': 0.739, 'danceability': 0.57, 'energy': 0.636, 'instrumentalness': 0.0221, 'key': 7, 'liveness': 0.0807, 'loudness': -8.136, 'mode': 1, 'speechiness': 0.0396, 'tempo': 101.984, 'valence': 0.219}, {'id': 'aade5391-0ae0-47d8-9681-63e9238c6407', 'href': 'https://open.spotify.com/track/4SqWKzw0CbA05TGszDgMlc', 'isrc': 'TCACC1438995', 'acousticness': 0.583, 'danceability': 0.575, 'energy': 0.648, 'instrumentalness': 0.0, 'key': 10, 'liveness': 0.115, 'loudness': -4.891, 'mode': 1, 'speechiness': 0.0358, 'tempo': 75.977, 'valence': 0.466}, {'id': '70ee017e-3936-4951-ad39-9e32d49edfe9', 'href': 'https://open.spotify.com/track/08TjqEEAO32VuF002ePbTz', 'isrc': 'TCADU1861917', 'acousticness': 0.0835, 'danceability': 0.259, 'energy': 0.521, 'instrumentalness': 0.0474, 'key': 8, 'liveness': 0.089, 'loudness': -8.022, 'mode': 1, 'speechiness': 0.0489, 'tempo': 75.003, 'valence': 0.267}, {'id': 'f19059af-5569-4453-b455-f17db97d1d87', 'href': 'https://open.spotify.com/track/0UV5zxRMz6AO4ZwUOZNIKI', 'isrc': 'USEP40937005', 'acousticness': 0.132, 'danceability': 0.454, 'energy': 0.82, 'instrumentalness': 0.000969, 'key': 2, 'liveness': 0.115, 'loudness': -4.193, 'mode': 1, 'speechiness': 0.0567, 'tempo': 166.303, 'valence': 0.575}, {'id': '5b54e267-7b83-4fa1-9729-303779ba25e5', 'href': 'https://open.spotify.com/track/7iN1s7xHE4ifF5povM6A48', 'isrc': 'GBAYE0601713', 'acousticness': 0.631, 'danceability': 0.443, 'energy': 0.403, 'instrumentalness': 0.0, 'key': 0, 'liveness': 0.111, 'loudness': -8.339, 'mode': 1, 'speechiness': 0.0322, 'tempo': 143.462, 'valence': 0.41}, {'id': 'caf171d8-543b-4c6c-9402-a5850354897c', 'href': 'https://open.spotify.com/track/6TvxPS4fj4LUdjw2es4g21', 'isrc': 'SEAYD8102080', 'acousticness': 0.796, 'danceability': 0.475, 'energy': 0.26, 'instrumentalness': 0.0016, 'key': 5, 'liveness': 0.109, 'loudness': -15.997, 'mode': 1, 'speechiness': 0.0322, 'tempo': 137.212, 'valence': 0.339}, {'id': '9728a30e-4184-4693-91cc-8642cd2e1155', 'href': 'https://open.spotify.com/track/4JGKZS7h4Qa16gOU3oNETV', 'isrc': 'USIR29300080', 'acousticness': 0.0031, 'danceability': 0.551, 'energy': 0.645, 'instrumentalness': 0.00376, 'key': 4, 'liveness': 0.421, 'loudness': -13.093, 'mode': 1, 'speechiness': 0.0354, 'tempo': 128.665, 'valence': 0.508}, {'id': '85aa6d37-3f77-46b8-ace1-406b1ccd16c3', 'href': 'https://open.spotify.com/track/2nilAlGEZmwyaLTMMyDdLo', 'isrc': 'US38Y0811508', 'acousticness': 0.487, 'danceability': 0.669, 'energy': 0.613, 'instrumentalness': 0.176, 'key': 4, 'liveness': 0.132, 'loudness': -11.12, 'mode': 0, 'speechiness': 0.036, 'tempo': 110.668, 'valence': 0.566}, {'id': '3266c023-1e0d-4a6e-a0d1-ec4082fbc399', 'href': 'https://open.spotify.com/track/0ASIqnVJvN1GmH1xEBdf2a', 'isrc': 'CAGOO1906218', 'acousticness': 0.417, 'danceability': 0.499, 'energy': 0.666, 'instrumentalness': 0.461, 'key': 2, 'liveness': 0.0715, 'loudness': -15.61, 'mode': 1, 'speechiness': 0.0446, 'tempo': 173.325, 'valence': 0.405}, {'id': 'd6a2693e-ad37-4027-b64d-88ea815f0f5d', 'href': 'https://open.spotify.com/track/0GegHVxeozw3rdjte45Bfx', 'isrc': 'USSUB0877702', 'acousticness': 0.44, 'danceability': 0.628, 'energy': 0.5, 'instrumentalness': 0.0, 'key': 6, 'liveness': 0.244, 'loudness': -9.66, 'mode': 0, 'speechiness': 0.0268, 'tempo': 124.932, 'valence': 0.68}, {'id': '2c95dc0e-031b-47f9-a758-87aeb2127534', 'href': 'https://open.spotify.com/track/2akMYW6w4sOWL1nhTzPJWu', 'isrc': 'US33X0907908', 'acousticness': 0.218, 'danceability': 0.565, 'energy': 0.716, 'instrumentalness': 3.3e-06, 'key': 10, 'liveness': 0.185, 'loudness': -6.254, 'mode': 1, 'speechiness': 0.0253, 'tempo': 92.977, 'valence': 0.679}, {'id': 'edc3b775-0319-4ba0-a4e2-e64fc655e37f', 'href': 'https://open.spotify.com/track/3HOXNIj8NjlgjQiBd3YVIi', 'isrc': 'USATO1400776', 'acousticness': 0.0948, 'danceability': 0.496, 'energy': 0.679, 'instrumentalness': 0.0, 'key': 10, 'liveness': 0.103, 'loudness': -7.898, 'mode': 1, 'speechiness': 0.0368, 'tempo': 154.028, 'valence': 0.507}, {'id': 'cf821d42-6bc3-470e-a2e9-4ced929b0be8', 'href': 'https://open.spotify.com/track/2H30WL3exSctlDC9GyRbD4', 'isrc': 'GBUM71603029', 'acousticness': 0.264, 'danceability': 0.496, 'energy': 0.644, 'instrumentalness': 0.0, 'key': 2, 'liveness': 0.0695, 'loudness': -5.385, 'mode': 1, 'speechiness': 0.0269, 'tempo': 96.017, 'valence': 0.435}, {'id': 'df747051-32c7-448e-a2a7-f91cbe12ad09', 'href': 'https://open.spotify.com/track/6tZetCGfhxPh5ZIKCGmaKq', 'isrc': 'USMRG1967001', 'acousticness': 0.134, 'danceability': 0.631, 'energy': 0.625, 'instrumentalness': 0.0196, 'key': 0, 'liveness': 0.117, 'loudness': -10.73, 'mode': 1, 'speechiness': 0.0374, 'tempo': 155.987, 'valence': 0.305}, {'id': 'dcff22e3-14ea-489a-bbca-e449c5245e9d', 'href': 'https://open.spotify.com/track/4sNG6zQBmtq7M8aeeKJRMQ', 'isrc': 'GBARL1500856', 'acousticness': 0.0941, 'danceability': 0.687, 'energy': 0.617, 'instrumentalness': 1.27e-05, 'key': 4, 'liveness': 0.0898, 'loudness': -5.213, 'mode': 1, 'speechiness': 0.0287, 'tempo': 121.079, 'valence': 0.665}, {'id': '080149fa-533a-4e4a-b59e-cea6428f9d55', 'href': 'https://open.spotify.com/track/1qbjCGGsZpEq56QgvtJoX7', 'isrc': 'USSM12102947', 'acousticness': 0.171, 'danceability': 0.646, 'energy': 0.611, 'instrumentalness': 0.0, 'key': 0, 'liveness': 0.0634, 'loudness': -6.516, 'mode': 1, 'speechiness': 0.0272, 'tempo': 109.051, 'valence': 0.272}, {'id': 'd9beca41-edb0-4e47-86bc-a8ffd31c6700', 'href': 'https://open.spotify.com/track/5kgyNmIytvTGGuiv0MwzZp', 'isrc': 'USUG11400492', 'acousticness': 0.432, 'danceability': 0.745, 'energy': 0.539, 'instrumentalness': 0.00162, 'key': 0, 'liveness': 0.0826, 'loudness': -4.41, 'mode': 1, 'speechiness': 0.0271, 'tempo': 95.893, 'valence': 0.886}]

    # scale song features from spotify/reccobeats calls
    track_features_df = pd.DataFrame(track_features)[FEATURES]

    scaled_feats = scaler.transform(track_features_df)
    distances, idxs = knn.kneighbors(scaled_feats)

    recs = df.iloc[idxs[0]]
    print(type(recs))
    print(recs)
    # print(recs[recs['name', 'artists']])
    # print(recs['name', 'artists'])
    # print(f"\nRecommendations for {track['name']}:")
    # print(df.iloc[idxs][['name', 'artists']])

def get_input(track_title, artist):
    pass

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


def main():
    recommend()

if __name__ == "__main__":
    main()