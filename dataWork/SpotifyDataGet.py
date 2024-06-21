# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('../Data/allSongs.csv', index_col=0)
data.Title = data['Title'].str.lower()
data.Artist = data['Artist'].str.lower()
data.head()

# %% [markdown]
# ### Spotify URI for songs

# %%
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm_notebook
import spotipy
import spotipy.util as util
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import time

# %%

client_id = '9d686c7d0833427db57420593ab7038b'

client_secret = '2e4ab33713e74b8990c8f1cf2da115f0'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# %%
titles = list(data.Title)
artists = list(data.Artist)
spotify_uri = list()
errors = list()
print(len(titles))

# %%
def get_spotify_uri(title, artist):
    title_clean = re.sub(r"[,.;@#?!&$%()]+", ' ', title)
    title_clean = re.sub('\s+', ' ', title_clean).strip()
    artist_clean = re.sub('\s+', ' ', artist).strip()
    
    query = title_clean + " " + artist_clean
    
    search = sp.search(q=query, limit=50, offset=0, type='track', market='US')
    search_items = search['tracks']['items']
    
    for i in range(len(search_items)):
        spotify_title = search_items[i]['name']
        spotify_artist = search_items[i]['artists'][0]['name']
        
        spotify_title_clean = re.sub(r"[,.;@#?!&$%()]+", ' ', spotify_title)
        spotify_title_clean = re.sub('\s+', ' ', title_clean).strip().lower()
        spotify_artist_clean = spotify_artist.lower().strip().lower()
        
        fuzzy_title_match = fuzz.token_set_ratio(title_clean, spotify_title_clean)
        fuzzy_artist_match = fuzz.token_set_ratio(artist_clean, spotify_artist_clean)
        fuzzy_match = (fuzzy_title_match + fuzzy_artist_match) / 2

        if (fuzzy_title_match >= 90) and (fuzzy_artist_match >= 50) and fuzzy_match >= 75:
            uri = search_items[i]['id']
            return uri
    return 0



# %%
temp = list()

for i in tqdm_notebook(range(len(titles))):
    uri = get_spotify_uri(titles[i], artists[i])
    
    if uri != 0:
        temp.append(uri)
    else:
        temp.append(uri)
        errors.append(i)
    print(uri)
spotify_uri = spotify_uri + temp

# %%
data['URI'] = spotify_uri
data = data[data.URI != 0]

# %%
data

# %%
data.to_csv('../Data/allSongs_w_uri.csv')

# %% [markdown]
# ### Get Spotify Features

# %%
data = pd.read_csv('../Data/allSongs_w_uri.csv', index_col=0)

# %%
uris = list(data.URI)

danceability_list = list()
energy_list = list()
key_list = list()
loudness_list = list()
mode_list = list()
speechiness_list = list()
acousticness_list = list()
instrumentalness_list = list()
liveness_list = list()
valence_list = list()
tempo_list = list()
duration_list = list()
time_signature_list= list()

# %%
def get_audio_features(uri):
    try:
        search = sp.audio_features(uri)
        if search[0] == None:
            danceability_list.append(np.nan)
            energy_list.append(np.nan)
            key_list.append(np.nan)
            loudness_list.append(np.nan)
            mode_list.append(np.nan)
            speechiness_list.append(np.nan)
            acousticness_list.append(np.nan)
            instrumentalness_list.append(np.nan)
            liveness_list.append(np.nan)
            valence_list.append(np.nan)
            tempo_list.append(np.nan)
            duration_list.append(np.nan)
            time_signature_list.append(np.nan) 
            return ('Error on: ' + str(uri))
        print(search)
        search_list = search[0]
        
        danceability_list.append(search_list['danceability'])
        energy_list.append(search_list['energy'])
        key_list.append(search_list['key'])
        loudness_list.append(search_list['loudness'])
        mode_list.append(search_list['mode'])
        speechiness_list.append(search_list['speechiness'])
        acousticness_list.append(search_list['acousticness'])
        instrumentalness_list.append(search_list['instrumentalness'])
        liveness_list.append(search_list['liveness'])
        valence_list.append(search_list['valence'])
        tempo_list.append(search_list['tempo'])
        duration_list.append(search_list['duration_ms'])
        time_signature_list.append(search_list['time_signature'])
    except Exception as e:
        if e.args[0]==429:
            raise Exception
        else:
            pass

# %%
for i in tqdm_notebook(range(len(uris)//100)):
    get_audio_features(uris[i:i+100])
get_audio_features(uris[17400:])

# %%
data['Danceability'] = danceability_list
data['Energy'] = energy_list
data['Key'] = key_list
data['Loudness'] = loudness_list
data['Mode'] = mode_list
data['Speechiness'] = speechiness_list
data['Acousticness'] = acousticness_list
data['Instrumentalness'] = instrumentalness_list
data['Liveness'] = liveness_list
data['Valence'] = valence_list
data['Tempo'] = tempo_list
data['Duration'] = duration_list
data['Time_Signature'] = time_signature_list

# %%
data = data.dropna()
data = data.drop_duplicates(subset="Title", keep="last")
data = data.reset_index(drop=True)
data.to_csv('songs_w_features.csv')

# %% [markdown]
# ### Get Year
# 

# %%
data = pd.read_csv("../Data/songs_w_features.csv", index_col=0)

# %%
songs = list(data.Title)
artists = list(data.Artist)
years = list()
errors = list()

# %%
def get_song_year(title, artist):
    title_clean = re.sub(r"[,.;@#?!&$%()]+", ' ', title)
    title_clean = re.sub('\s+', ' ', title_clean).strip()
    artist_clean = re.sub('\s+', ' ', artist).strip()
    
    query = title_clean + " " + artist_clean
    
    try:
        search = sp.search(q=query, limit=50, offset=0, type='track')
        search_items = search['tracks']['items']
        year = search_items[0]['album']['release_date']
        return year
    except Exception:
        year = 0
        return year

# %%
for i in tqdm_notebook(range(len(songs))):
    year = get_song_year(songs[i], artists[i])
    
    if year != 0 :
        years.append(year)
    else:
        years.append(year)
        print("Errored on " + str(i))
        errors.append(i)    

# %%
new_years = list()

def clean_year(date):
    y = date.split('-')[0]
    return int(y)

# %%
for i in years:
    if i == 0:
        new_years.append(i)
    else:
        x = clean_year(i)
        new_years.append(x)

# %%
data['Release_Year'] = new_years
data = data[data.Release_Year != 0]

# %%
data.to_csv('../Data/songs_w_features_year.csv')
data


