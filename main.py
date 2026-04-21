from CollaborativeRecommender import CollaborativeRecommender
import pandas as pd
from scipy import sparse

# Nick's model
def collaborative_exp(song, artist, n):
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


    # Instantiate and fit the model
    model = CollaborativeRecommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features,
                        decode_id_song=decode_id_song)

    model.decode_id_artists = decode_id_artists
    model.decode_id_title_artist = decode_id_title_artist

    input_string = f"{song} - {artist}"

    new_recommendations = model.make_recommendation(new_song=input_string, n_recommendations=n)


    print(f"Recommendations for {song} by {artist}:")
    print("\n".join(new_recommendations))


def main():
    N = 1
    SONG = 'Television Rules the Nation / Creshendolls'
    ARTIST = 'Daft Punk'

    collaborative_exp(SONG, ARTIST, N)

    done = False

    while not done:
        song = input("Enter song name: ")
        artist = input("Enter artist name: ")
        collaborative_exp(song, artist, N)

        cont = input("Continue? (y/n): ")
        if cont != "y":
            done = True

if __name__ == "__main__":
    main()