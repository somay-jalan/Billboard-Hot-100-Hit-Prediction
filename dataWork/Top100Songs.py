# %%
import pandas as pd
import numpy as np

top100Songs = pd.DataFrame(columns=['Title', 'Artist', 'Year'])

url_template = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'

years = [i for i in range(1959,2024)]

# %%
for i in years:
    try:
        url = url_template + str(i)
        if i ==2012 or i==2013 or i==2023:
            temp = pd.read_html(url, header=0)[1]
        else:
            temp = pd.read_html(url, header=0)[0]
        temp.head()
        temp.columns.values[2] = "Artist"
        temp = temp.drop(temp.columns[0], axis=1)
        temp.Artist = [artist.split(' featuring')[0] for artist in temp.Artist]
        temp.Title = [title.strip('\"') for title in temp.Title]
        temp['Year'] = i
        top100Songs = top100Songs._append(temp)
    except:
        print(temp)



# %%
top100Songs = top100Songs.reset_index(drop=True)
top100Songs['Year'] = pd.to_numeric(top100Songs['Year'])
top100Songs['Title'] = top100Songs['Title'].astype(str)
top100Songs['Artist'] = top100Songs['Artist'].astype(str)
top100Songs

# %%
top100Songs.to_csv('../Data/top100.csv')


