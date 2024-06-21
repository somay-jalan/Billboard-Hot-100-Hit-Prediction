# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# %%
data = pd.read_csv('../Data/songs_w_features_year.csv', index_col=0)

# %%
fig1 = sns.catplot(x="Top100", kind="count", data=data)
fig1.set(title='Distribution of Billboard Top 100 Songs')
fig1.savefig("../images/data-distribution.png")
plt.show()

# %%
decade_50 = data[(data.Release_Year >= 1950) & (data.Release_Year < 1960)].shape[0]
decade_60 = data[(data.Release_Year >= 1960) & (data.Release_Year < 1970)].shape[0]
decade_70 = data[(data.Release_Year >= 1970) & (data.Release_Year < 1980)].shape[0]
decade_80 = data[(data.Release_Year >= 1980) & (data.Release_Year < 1990)].shape[0]
decade_90 = data[(data.Release_Year >= 1990) & (data.Release_Year < 2000)].shape[0]
decade_00 = data[(data.Release_Year >= 2000) & (data.Release_Year < 2010)].shape[0]
decade_10 = data[(data.Release_Year >= 2010) & (data.Release_Year < 2019)].shape[0]
decade_20 = data[(data.Release_Year >= 2019) & (data.Release_Year < 2029)].shape[0]

# %%
decades = ['1950s','1960s','1970s','1980s','1990s', '2000s', '2010s','2020s']
decade_frq = [decade_50,decade_60,decade_70,decade_80,decade_90, decade_00, decade_10,decade_20]


# %%
fig2 = sns.barplot(x=decades, y=decade_frq)
fig2.set(xlabel='Decade', ylabel='Frequency', title='Frequency vs. Decade')
fig2.figure.savefig("../images/fig-vs-decade.png")

# %%
fig3 = sns.relplot(x="Release_Year", y="Danceability", hue="Top100", kind="line", data=data, height=5, aspect=3)
fig3.set(title='Danceability vs. Time')
fig3.savefig('../images/dance-vs-time.png')

# %%
fig4 = sns.relplot(x="Release_Year", y="Energy", hue="Top100", kind="line", data=data, height=5, aspect=3)
fig4.set(title='Energy vs. Time')
fig4.savefig('../images/energy-vs-time.png')

# %%
fig5 = sns.relplot(x="Release_Year", y="Loudness", hue="Top100", kind="line", data=data, height=5, aspect=2)
fig5.set(title='Loudness vs. Time')
fig5.savefig('../images/loud-vs-time.png')

# %%
fig6 = sns.relplot(x="Release_Year", y="Speechiness", hue="Top100", kind="line", data=data, height=5, aspect=2)
fig6.set(title='Speechiness vs. Time')
fig6.savefig('../images/speech-vs-time.png')

# %%
fig7 = sns.relplot(x="Release_Year", y="Acousticness", hue="Top100", kind="line", data=data, height=5, aspect=2)
fig7.set(title='Acousticness vs. Time')
fig7.savefig('../images/acoustic-vs-time.png')

# %%
fig8 = sns.relplot(x="Release_Year", y="Liveness", hue="Top100", kind="line", data=data, height=5, aspect=2)
fig8.set(title='Liveness vs. Time')
fig8.savefig('../images/live-vs-time.png')

# %%
plt.figure(figsize=(12, 8))
numerical_cols = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


