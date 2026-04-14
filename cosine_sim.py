import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import requests
import http.client
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    'energy',
    'danceability',
    'tempo',
    'acousticness',
    'instrumentalness',
    'liveness',
    'loudness',
    'speechiness',
    'valence'
]

conn = http.client.HTTPSConnection("api.reccobeats.com")
payload = ''
headers = {
  'Accept': 'application/json'
}

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


def top_20_with_info():
    top_tracks = get_top_spotify_tracks()
    track_features = get_tracks_info(top_tracks)
    for tf in track_features:
        print(tf)

    df = pd.DataFrame(track_features)
    print(df.info)

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
if __name__ == "__main__":
    cosine_sim()