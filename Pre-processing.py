import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from scipy import sparse

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import requests


LEAST_SONGS = 100
NUM_COLS = 1000
FRAC = .1

music_info = pd.read_csv("data/Music_Info.csv",
                        sep=',')
user_info = pd.read_csv("data/User Listening History.csv",
                        sep=',')


# merge/clean datasets

# select users who have listened to at least 16 songs FIRST
user_counts = user_info['user_id'].value_counts()
valid_users = user_counts[user_counts >= 16].index

# filter before merge
user_info_filtered = user_info[user_info['user_id'].isin(valid_users)]

# merge smaller
songs = pd.merge(user_info_filtered, music_info, on="track_id", how="left")

# remove duplicates AFTER merge
songs.drop_duplicates(['user_id', 'track_id'], inplace=True)


# select users who have listened to at least 16 songs (number can be changed)
song_per_user = songs.groupby('user_id')['track_id'].count()
song_16_id = song_per_user[song_per_user > LEAST_SONGS].index.to_list()
df_song_id_more_16 = songs[songs['user_id'].isin(song_16_id)].reset_index(drop=True)

# select users who have listened to at least 16 songs (number can be changed)
song_per_user = songs.groupby('user_id')['track_id'].count()
song_16_id = song_per_user[song_per_user > LEAST_SONGS].index.to_list()
df_song_id_more_16 = songs[songs['user_id'].isin(song_16_id)].reset_index(drop=True)

songs_sample = df_song_id_more_16


# convert dataframe to sparce matrix
songs_sample['track_id'] = songs_sample['track_id'].astype('category')
songs_sample['user_id'] = songs_sample['user_id'].astype('category')

df_songs_features = songs_sample.pivot_table(
    index='track_id',
    columns='user_id',
    values='playcount',
    fill_value=0
)

mat_songs_features = csr_matrix(df_songs_features.values)




# list of unique songs
df_unique_songs = music_info.drop_duplicates(subset='track_id').set_index('track_id')

song_metadata = music_info.set_index('track_id').loc[df_songs_features.index][['name', 'artist']]

song_metadata.to_csv('data/song_metadata.csv', index=False)
sparse.save_npz('data/mat_song_features.npz', mat_songs_features)
