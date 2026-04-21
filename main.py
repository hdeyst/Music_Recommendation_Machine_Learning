from CollaborativeRecommender import collaborative_exp
from clustering import recommend
from cosine_sim import cos_sim


def main():
    N = 1
    SONG = 'Television Rules the Nation / Creshendolls'
    ARTIST = 'Daft Punk'

    # cos_sim(SONG, ARTIST)
    collaborative_exp(SONG, ARTIST, N)
    # recommend(SONG, ARTIST)

    done = False

    while not done:
        song = input("Enter song name: ")
        artist = input("Enter artist name: ")
        # cos_sim(song, artist)
        collaborative_exp(song, artist, N)
        # recommend(song, artist)

        cont = input("Continue? (y/n): ")
        if cont != "y":
            done = True

if __name__ == "__main__":
    main()