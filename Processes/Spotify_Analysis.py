import pandas as pd
from User import User, see_all

def new_spotify_df():

    SPOTIPY_CLIENT_ID = "c833d78c87e24cbe801c41028494f694"
    SPOTIPY_CLIENT_SECRET = "f86a3cd9e87b486a9357866d962d7587"

    client_1 = User(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)

    target_playlist_id = ["5tLIjF9PAYjAs0nqdtR27T", "4GDMzc7TUhxmXwhm8A94VY", "6MpWr9B1EzssluJp65Ok8Z", "583F7R73ptV2qyzVYqdsQC"]

    client_1.target_playlist(target_playlist_id)

    client_1.target_playlist_df = client_1.analysis(client_1.target_playlist_df)
    target_playlist = client_1.target_playlist_df

    target_playlist.to_csv("C:\\Users\\diogo\\Desktop\\perkier tech\\Shuffle\\data\\target_playlist_1.csv", index=False)

    client_1.get_playlist()
    client_1.playlist_df = client_1.analysis(client_1.playlist_df)
    playlist_df = client_1.playlist_df

    playlist_df.to_csv("C:\\Users\\diogo\\Desktop\\perkier tech\\Shuffle\\data\\playlist_1.csv", index=False)

    return playlist_df, target_playlist

def import_spotify_df(playlist_path, target_path):

    playlist_df = pd.read_csv(playlist_path)
    target_playlist = pd.read_csv(target_path)

    return playlist_df, target_playlist

def main():

    # Create new spotify DataFrames from scratch
    # playlist_df, target_playlist = new_spotify_df()

    # Import previous spotify DataFrames
    playlist_df, target_playlist = import_spotify_df("C:\\Users\\diogo\\Desktop\\perkier tech\\Shuffle\\data\\playlist_1.csv",
                                                     "C:\\Users\\diogo\\Desktop\\perkier tech\\Shuffle\\data\\target_playlist_1.csv")

    playlist_df = playlist_df.loc[playlist_df["duration_s"] != " "].reset_index(drop=True)

    import Data_Pre_Analysis as dpa

    clean_dataset = dpa.Clean_Data(playlist_df)
    playlist_df = clean_dataset.df

    target_clean_dataset = dpa.Clean_Data(target_playlist)
    target_playlist = target_clean_dataset.df

    # target_processing = dpa.Pre_Processing(target_playlist)
    # target_processing.dist_analysis()

    import Data_Dive as dd

    dd.correlation_plots(target_playlist, ['energy', 'danceability', 'valence', 'tempo_bpm'])

    dd.rad(target_playlist, 'playlist_name', ['energy', 'danceability', 'valence', 'tempo_bpm'])

    dd.check_randomness(target_playlist, "valence")
    dd.check_randomness(target_playlist, "energy")
    dd.check_randomness(target_playlist, "danceability")

    dd.dist_analysis(target_playlist, "valence")
    dd.dist_analysis(target_playlist, "energy")
    dd.dist_analysis(target_playlist, "danceability")

    quantitative_df = playlist_df.drop(['track_id', 'playlist_name', 'playlist_creator', 'artist_genres'], axis=1)

    pre_processing = dpa.Pre_Processing(playlist_df)

    pre_processing.dist_analysis()


    # pre_processing.plot_missing_data()
    #
    # pre_processing.pair_plt(quantitative_df)
    #
    # pre_processing.get_correlations()
    # pre_processing.correlation.plot()


    # pre_processing.id_outliers()

    quit()


if __name__ == '__main__':

    see_all()
    main()
