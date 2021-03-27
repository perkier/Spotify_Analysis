# Data management Libraries
import pandas as pd
import numpy as np

# Import spotipy library
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

# Utilities
from time import sleep
import ast


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)


class User:

    def __init__(self, client_ID, client_secret):

        self.id = client_ID
        self.secret = client_secret

        self.renew_sp()

    def renew_sp(self):

        scope = 'playlist-read-private'

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.id,
                                                            client_secret=self.secret,
                                                            redirect_uri="http://localhost:8888/callback",
                                                            scope=scope))

    def read_playlist(self, playlist_df, results):

        # Loops through all playlists saved by the user
        for i, item in enumerate(results):
            # Permission to access user playlists
            response = self.sp.playlist_items(item['uri'],
                                              offset=0,
                                              fields='items.track.id,total',
                                              additional_types=['track'])

            # Converts tracks of playlist dictionary to dataframe
            playlist_tracks = pd.DataFrame.from_dict(response["items"])

            # track_id dictionary to string
            playlist_tracks["track_id"] = playlist_tracks["track"].apply(lambda x: f"spotify:track:{x['id']}")
            playlist_tracks = playlist_tracks.drop(["track"], axis=1)

            # Adds playlist name and creator to the dataframe
            playlist_tracks["playlist_name"] = item['name']
            playlist_tracks["playlist_creator"] = item['owner']['display_name']

            # Concats current loop playlist dataframe with main playlists dataframe
            playlist_df = pd.concat([playlist_df, playlist_tracks]).reset_index(drop=True)

        return playlist_df

    def get_playlist(self):

        results = self.sp.current_user_playlists(limit=50)

        # Creates empty dataframe to be populated by user tracks in playlist
        playlist_df = pd.DataFrame()

        playlist_df = self.read_playlist(playlist_df, results['items'])

        self.playlist_df = playlist_df


    def target_playlist(self, playlist_id):

        # Creates empty dataframe to be populated by user tracks in playlist
        self.target_playlist_df = pd.DataFrame()

        for plyst in playlist_id:

            results = self.sp.playlist(plyst, fields=None, market=None, additional_types=('track', ))

            self.target_playlist_df = self.read_playlist(self.target_playlist_df, [results])


    def level_2_info(self, track_urn):
        # Performs a more indepth analysis of a song

        self.counter_error = 0

        try:

            analysis = self.sp.audio_analysis(track_urn)
            track = self.sp.track(track_urn)
            track_features = self.sp.audio_features(track_urn)

            duration_s = analysis["track"]["duration"]
            loudness_db = analysis["track"]["loudness"]
            tempo_bpm = analysis["track"]["tempo"]
            time_signature = analysis["track"]["time_signature"]
            key = analysis["track"]["key"]
            mode = analysis["track"]["mode"]

            danceability = track_features[0]["danceability"]
            energy = track_features[0]["energy"]
            speechiness = track_features[0]["speechiness"]
            acousticness = track_features[0]["acousticness"]
            instrumentalness = track_features[0]["instrumentalness"]
            liveness = track_features[0]["liveness"]
            valence = track_features[0]["valence"]

            explicit = track["explicit"]
            popularity = track["popularity"]

            album_id = track['album']['id']
            album = self.sp.album(album_id)
            album_genres = album["genres"]

            artist_id = [artist['id'] for artist in track['artists']]
            artist = self.sp.artist(artist_id[0])

            artist_genres = artist['genres']
            artist_followers = artist['followers']['total']
            artist_popularity = artist['popularity']

        except:

            self.renew_sp()

            duration_s = " "
            loudness_db = " "
            tempo_bpm = " "
            time_signature = " "
            key = " "
            mode = " "

            danceability = " "
            energy = " "
            speechiness = " "
            acousticness = " "
            instrumentalness = " "
            liveness = " "
            valence = " "

            explicit = " "
            popularity = " "

            artist_genres = " "
            artist_followers = " "
            artist_popularity = " "

            if self.counter_error == (self.counter - 1):

                sleep(10)
                print("Error occured, I'll take a 10 second nap")

            else:
                print("Error occured, I'll renew the permissions")

            self.counter_error = self.counter

        self.counter += 1

        return f"{duration_s}//{loudness_db}//{tempo_bpm}//{time_signature}//{key}//{mode}//{danceability}//{energy}//{speechiness}//{acousticness}//{instrumentalness}//{liveness}//{valence}//{explicit}//{popularity}//{artist_popularity}//{artist_genres}//{artist_followers}"


    def analysis(self, playlist_df):
        # Analysis of every song from a given playlist

        self.counter = 0
        while_counter = 0

        # Search for the missing tracks of the playlist and deletes them
        playlist_df = playlist_df.loc[playlist_df["track_id"] != "spotify:track:None"].reset_index(drop=True)

        playlist_df["analysis"] = playlist_df["track_id"].apply(lambda x: self.level_2_info(x))

        playlist_df[["duration_s", "loudness_db", "tempo_bpm",
                     "time_signature", "key", "mode",
                     "danceability", "energy", "speechiness",
                     "acousticness", "instrumentalness",
                     "liveness", "valence", "explicit",
                     "popularity", "artist_popularity",
                     "artist_genres", "artist_followers"]] = playlist_df['analysis'].str.split('//', expand=True)

        playlist_df = playlist_df.drop(["analysis"], axis=1)

        reloop_data = playlist_df.loc[playlist_df["duration_s"] == " "].reset_index(drop=True)

        if len(reloop_data) > 0:

            while len(reloop_data) > 0:

                level_2_data = playlist_df.loc[playlist_df["duration_s"] != " "].reset_index(drop=True)
                reloop_data = playlist_df.loc[playlist_df["duration_s"] == " "].reset_index(drop=True)

                reloop_data = reloop_data[['track_id', 'playlist_name', 'playlist_creator']]

                reloop_data["analysis"] = reloop_data["track_id"].apply(lambda x: self.level_2_info(x))

                reloop_data[["duration_s", "loudness_db", "tempo_bpm",
                             "time_signature", "key", "mode",
                             "danceability", "energy", "speechiness",
                             "acousticness", "instrumentalness",
                             "liveness", "valence", "explicit",
                             "popularity", "artist_popularity",
                             "artist_genres", "artist_followers"]] = reloop_data['analysis'].str.split('//',expand=True)

                reloop_data = reloop_data.drop(["analysis"], axis=1)

                playlist_df = pd.concat([level_2_data, reloop_data]).reset_index(drop=True)

                while_counter += 1
                print(while_counter)
                print(reloop_data)

                if while_counter > 10:
                    break

        playlist_df = playlist_df.loc[playlist_df["duration_s"] != " "].reset_index(drop=True)

        return playlist_df
