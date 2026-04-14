import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import json
import time
import requests
import ast
SONGS = pd.read_csv("data/tracks_features.csv")
## Ask for user input
def create_song_df():
    song_df = None
    valid_choice = False
    while not valid_choice:
        temp = input('Do you want to analyze A) One song or B) All your data?')
        if temp.lower() == 'a':
            while not valid_choice:
                song = input('Please enter the name of the song:')
                song_df = SONGS[SONGS['name'].str.contains(song, case=False)]
                artist_name = input('Please enter the name of the artist:')
                song_df = song_df[song_df['artists'].str.contains(artist_name, case=False)]
                if song_df.empty:
                    print('No song or artist names in dataframe. Please try again.')
                elif len(song_df) > 1:
                    while not valid_choice:
                        print('More than one match in the dataframe. Please select from the following options:')
                        for i, row in enumerate(song_df.itertuples(index=False), start=1):
                            print(f"{i}: {row.name} by {row.artists}")
                        user_selection = int(input('Please select the number of the correct song above: '))
                        if user_selection > len(song_df) or user_selection < 1:
                            print('Invalid input. Please try again.')
                        else:
                            song_df = song_df.iloc[[user_selection - 1]]
                            valid_choice = True
                else:
                    valid_choice = True
            print(f"Getting a recommendation for {song_df['name'].iloc[0]} by {song_df['artists'].iloc[0]}")
        elif temp.lower() == 'b':
            valid_choice = False
            while not valid_choice:
                file_name = input('Please enter the name of the file to read: ')
                if os.path.exists(file_name):
                    song_df = pd.read_csv(file_name)
                    # Cleaning dataframe and summarizing by count
                    df["count"] = 1
                    df = (
                        df.groupby(["artist", "track"], as_index=False)["count"]
                        .sum()
                        .sort_values("count", ascending=False)
                        .reset_index(drop=True)
                    )
                    if len(song_df) > 200:
                        song_df = song_df.iloc[:200]    # Trimming df if it's longer than 200 values
                    obtain_features(song_df)
                    valid_choice = True
                else:
                    print('File does not exist. Please try again.')
        else:
            print('Invalid input. Please try again.')
    return song_df

def extract_first_artist(x):
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x[0] if len(x) > 0 else None
    if isinstance(x, str):
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
    return None

def obtain_features(df):
    # Loading credentials
    with open("credentials.json", "r") as f:
        creds = json.load(f)

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=creds["CLIENT_ID"],
            client_secret=creds["CLIENT_SECRET"]
        )
    )

    # Calling spotify API to get spotify IDs
    spotify_ids = []
    print("Searching Spotify for IDs...")
    for i, row in df.iterrows():
        artist = row["artist"]
        track = row["track"]
        spotify_id = search_track_id(artist, track, sp)

        spotify_ids.append(spotify_id)

        if (i + 1) % 25 == 0 or i == len(df) - 1:
            print(f"Processed {i + 1}/{len(df)}")

        time.sleep(0.1)

    df["spotify_id"] = spotify_ids

    all_features = []

    # Getting the features from Reccobeats API
    for i in range(0, len(spotify_ids), 40):
        batch = spotify_ids[i:i + 40]

        response = requests.get(
            "https://api.reccobeats.com/v1/audio-features",
            params={"ids": ",".join(batch)},
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()

        batch_features = response.json().get("content", [])
        all_features.extend(batch_features)

        print(f"Fetched features for batch {i // 41} ({len(batch)} songs)")
        time.sleep(0.1)

    return all_features

def create_feature_df(df, features):
    features_df = pd.DataFrame(features)
    if "spotify_id" not in features_df.columns:
        features_df["spotify_id"] = features_df["href"].str.extract(r"/track/([^/?]+)")

    # Keep only the columns you want
    features_df = features_df[[
        "spotify_id",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "valence"
    ]]

    # Merge with original df
    merged_df = df.merge(features_df, on="spotify_id", how="left")

    return merged_df

def search_track_id(artist, track, sp):
    query = f'track:"{track}" artist:"{artist}"'
    result = sp.search(q=query, type="track", limit=1)

    items = result.get("tracks", {}).get("items", [])
    if not items:
        return None

    return items[0]["id"]