import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from scipy import sparse

class CollaborativeRecommender:
    '''
    Initializes the recommendation model
    metric: the distance metric we use (cosine)
    algorithm: algorithm used to compute the nearest neighbors (brute)
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
            n_neighbors=n_recommendations + 1
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
        i: f"{title} by {artist}"
        for i, (title, artist) in decode_id_artists.items()
    }


    # Instantiate and fit the model
    model = CollaborativeRecommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features,
                        decode_id_song=decode_id_song)

    model.decode_id_artists = decode_id_artists
    model.decode_id_title_artist = decode_id_title_artist

    input_string = f"{song} by {artist}"

    new_recommendations = model.make_recommendation(new_song=input_string, n_recommendations=n)

    # print(f"Recommendations for {song} by {artist}:")
    print("\n".join(new_recommendations))
