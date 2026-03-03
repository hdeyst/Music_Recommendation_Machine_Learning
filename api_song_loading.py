import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

import http.client

conn = http.client.HTTPSConnection("api.reccobeats.com")
payload = ''
headers = {
  'Accept': 'application/json'
}

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


top_tracks = sp.current_user_top_artists(time_range='long_term', limit=20)

print("Your Top Tracks:")
for i, track in enumerate(top_tracks['items']):
    #print(track['id'])
    artist_name = track['artists'][0]['name']
    track_name = track['name']


 
    #features = GET /audio-features/{track['id']}
    print(f"{i+1}. {artist_name} - {track_name}")