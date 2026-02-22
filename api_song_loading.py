import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

CLIENT_ID     = "7e3a150dea7f4dd3a7662f87be608726"
CLIENT_SECRET = "05fe1f38ef1d4261b61b313c82029d42"
REDIRECT_URI  = "http://127.0.0.1:8888/callback"
SCOPE         = "user-top-read" #will have to learn the options

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
))

top_tracks = sp.current_user_top_tracks(time_range='long_term', limit=20)

print("Your Top Tracks:")
for i, track in enumerate(top_tracks['items']):
    artist_name = track['artists'][0]['name']
    track_name = track['name']
    print(f"{i+1}. {artist_name} - {track_name}")