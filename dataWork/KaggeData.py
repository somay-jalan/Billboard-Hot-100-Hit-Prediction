# %%
import pandas as pd

# %%
spotify_songs = pd.read_csv('../Data/KaggleData/song_info.csv')
spotify_songs

# %%
spotify_songs = spotify_songs.drop('album_names',axis=1)
spotify_songs = spotify_songs.drop('playlist',axis=1)

spotify_songs = spotify_songs.rename(columns={"song_name": "Title", "artist_name": "Artist"})

spotify_songs = spotify_songs.drop_duplicates(subset="Title", keep="first")
spotify_songs

# %% [markdown]
# ### Combining Spotify Data and Top 100 data
# 

# %%
top100 = pd.read_csv('../Data/top100.csv', index_col=0)
top100

# %%
top100 = top100.drop('Year', axis=1)
top100 = top100.drop_duplicates(subset="Title", keep="first")
top100

# %%
spotify_songs['Top100'] = 0
top100['Top100'] = 1
spotify_songs

# %%
combined = pd.concat([spotify_songs, top100], ignore_index=True, sort=True)
combined = combined.drop_duplicates(subset="Title", keep="last")
combined = combined.reset_index(drop=True)
combined

# %%
combined.to_csv('../Data/allSongs.csv')


