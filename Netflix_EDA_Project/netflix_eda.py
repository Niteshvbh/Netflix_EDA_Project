"""
Created on Thu Jul 31 18:25 2025

@author: Nitesh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

# Load the dataset
df = pd.read_csv(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\data\netflix_titles.csv", encoding = "latin1")

# Basic info
print(df.info())

# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'])
df['date_added'] = df['date_added'].fillna('2000-01-01')
df['month_added'] = df['date_added'].dt.month_name()
df['weekday_added'] = df['date_added'].dt.day_name()
df['year_added'] = df['date_added'].dt.year

# Split 'duration' into two new columns: 'duration_int' and 'duration_type'
df[['duration_int', 'duration_type']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')
df['duration_int'] = pd.to_numeric(df['duration_int'])

# Fill missing 'director', 'cast', 'country' with 'Unknown'
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
country_data = df['country'].dropna()

# Drop rows with missing 'rating', 'duration_int', or 'duration_type' (only a few rows)
df.dropna(subset=['rating', 'duration_int', 'duration_type'], inplace=True)

# Clean text columns (optional polishing)
text_cols = ['type', 'title', 'director', 'cast', 'country', 'rating', 'duration_type', 'listed_in']
for col in text_cols:
    df[col] = df[col].str.strip()         # remove leading/trailing spaces
    df[col] = df[col].str.replace('\n', ' ', regex=True)  # clean line breaks if any

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Check remaining nulls
print("Remaining nulls:\n", df.isnull().sum())

# Split and flatten all genres
genre_list = []
for item in df['listed_in']:
    genre_list.extend([genre.strip() for genre in item.split(',')])

# Count top genres
top_genres = Counter(genre_list).most_common(10)
genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

print(genres_df)

type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

fig = px.pie(type_counts, names='Type', values='Count',
             title="Distribution of Movies vs TV Shows",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()

movies_df = df[df['type'] == 'Movie']  # ✅ Define movies_df
tv_df = df[df['type'] == 'TV Show']

plt.figure(figsize=(10, 4))
sns.histplot(data=movies_df, x='release_year', bins=50, color='skyblue')  # ✅ Now this will work
plt.title("Movies Released per Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\movies_release_per_year.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x='year_added', data=df, order=sorted(df['year_added'].dropna().unique()), 
              color='salmon')
plt.title("Content Added to Netflix by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Content Added to Netflix by Year.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Split countries and flatten
countries = []
for entry in country_data:
    countries.extend([x.strip() for x in entry.split(',')])

# Count top 10
top_countries = Counter(countries).most_common(10)
countries_df = pd.DataFrame(top_countries, columns=['Country', 'Count'])

plt.figure(figsize=(10, 4))
sns.barplot(data=countries_df, x='Country', y='Count', color='lightgreen')
plt.title("Top 10 Countries by Content Count")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.xlabel("Country")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 Countries by Content Count.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Exclude 'Unknown'
directors = df[df['director'] != 'Unknown']['director']
top_directors = directors.value_counts().head(10)

plt.figure(figsize=(10, 4))
sns.barplot(x=top_directors.values, y=top_directors.index, color='lightblue')  # Again, use color
plt.title("Top 10 Directors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 Directors on Netflix.png",
            dpi=300, bbox_inches='tight')
plt.show()

genre_list = []

for item in df['listed_in']:
    genre_list.extend([genre.strip() for genre in item.split(',')])

top_genres = Counter(genre_list).most_common(10)
genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

# Genre analysis already done previously
fig = px.bar(genres_df, x='Genre', y='Count',
             title="Top 10 Netflix Genres (Interactive)",
             labels={'Count': 'Number of Titles'},
             color='Genre',
             template='plotly',
             text='Count')

fig.update_traces(textposition='outside')
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Check unique ratings
print("Unique ratings:")
print(df['rating'].value_counts())

# Count ratings
rating_counts = df['rating'].value_counts()

# Define a threshold — if a rating appears less than 100 times, group it as "Others"
threshold = 100
df['rating_cleaned'] = df['rating'].apply(lambda x: x if rating_counts[x] >= threshold else 'Others')

# Confirm new cleaned rating categories
print(df['rating_cleaned'].value_counts())

# Set figure size
plt.figure(figsize=(10, 4))

sns.countplot(data=df, x='rating_cleaned', order=df['rating_cleaned'].value_counts().index, color='Blue')

plt.title("Distribution of Netflix Content by Rating")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Distribution of Netflix Content by Rating.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(movies_df['duration_int'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Distribution of Movie Durations.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x='duration_int', data=tv_df, order=sorted(tv_df['duration_int'].unique()), color='salmon')
plt.title("Number of Seasons in TV Shows")
plt.xlabel("Seasons")
plt.ylabel("Number of TV Shows")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Number of Seasons in TV Shows.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
order = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']
sns.countplot(x='month_added', data=df, order=order, color='mediumseagreen')
plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Number of Titles Added")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Content Added by Month.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(x='weekday_added', data=df, order=order, color='coral')
plt.title("Content Added by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Number of Titles Added")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Content Added by Day of the Week.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index, color='mediumpurple')
plt.title("Distribution of Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Distribution of Content Ratings.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index, color='pink')
plt.title("Content Rating by Type")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Content Rating by Type.png",
            dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=genres_df, x='Genre', y='Count', color='orchid')
plt.title("Top 10 Most Common Genres on Netflix")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 Most Common Genres on Netflix.png",
            dpi=300, bbox_inches='tight')
plt.show()

fig = px.bar(genres_df, x='Genre', y='Count',
             title="Top 10 Netflix Genres (Interactive)",
             labels={'Count': 'Number of Titles'},
             color='Genre',
             template='plotly',
             text='Count')

fig.update_traces(textposition='outside')
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Filter genres separately for Movies and TV Shows
movie_genres = []
for item in df[df['type'] == 'Movie']['listed_in']:
    movie_genres.extend([genre.strip() for genre in item.split(',')])

tv_genres = []
for item in df[df['type'] == 'TV Show']['listed_in']:
    tv_genres.extend([genre.strip() for genre in item.split(',')])

# Count top 5 in each
top_movie_genres = Counter(movie_genres).most_common(5)
top_tv_genres = Counter(tv_genres).most_common(5)

print("Top Movie Genres:", top_movie_genres)
print("Top TV Show Genres:", top_tv_genres)

# Compare ratings by type (Movie vs TV Show)
plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='rating_cleaned', hue='type', 
              order=df['rating_cleaned'].value_counts().index)
plt.title("Ratings Distribution by Content Type")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Type")
plt.tight_layout()
plt.grid(axis='y')
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Ratings Distribution by Content Type.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Split genre strings into lists
genre_series = df['listed_in'].dropna().apply(lambda x: [genre.strip() for genre in x.split(',')])

# Flatten the list
all_genres = [genre for sublist in genre_series for genre in sublist]

# Get top 10
top_genres = Counter(all_genres).most_common(10)

# Convert to DataFrame
genre_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

plt.figure(figsize=(10, 5))
sns.barplot(data=genre_df, x='Genre', y='Count', color='skyblue')
plt.title("Top 10 Genres on Netflix")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 Genres on Netflix.png",
            dpi=300, bbox_inches='tight')
plt.show()

# ✅ Extract primary genre
df['primary_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

movies_df = df[df['type'] == 'Movie']
tv_df = df[df['type'] == 'TV Show']

# ✅ Count top genres
movie_genres = movies_df['primary_genre'].value_counts().head(10)
tv_genres = tv_df['primary_genre'].value_counts().head(10)

# ✅ Plot side-by-side bar charts
plt.figure(figsize=(14, 6))

# Movies
plt.subplot(1, 2, 1)
sns.barplot(x=movie_genres.values, y=movie_genres.index, color='Blue')
plt.title("Top 10 Movie Genres")
plt.xlabel("Count")
plt.ylabel("Genre")

# TV Shows
plt.subplot(1, 2, 2)
sns.barplot(x=tv_genres.values, y=tv_genres.index, color='Purple')
plt.title("Top 10 TV Show Genres")
plt.xlabel("Count")
plt.ylabel("Genre")

plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 TV Show Genres.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Combine all descriptions per type
movie_text = " ".join(df[df['type'] == 'Movie']['description'].dropna().tolist())
tv_text = " ".join(df[df['type'] == 'TV Show']['description'].dropna().tolist())

# Define stopwords (commonly excluded words)
stopwords = set(STOPWORDS)

# Generate word clouds
movie_wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(movie_text)
tv_wc = WordCloud(width=800, height=400, background_color='black', colormap='Set2', stopwords=stopwords).generate(tv_text)

# Plot Word Cloud for Movies
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(movie_wc, interpolation='bilinear')
plt.axis('off')
plt.title("🎬 Word Cloud - Movies Descriptions", fontsize=14)

# Plot Word Cloud for TV Shows
plt.subplot(1, 2, 2)
plt.imshow(tv_wc, interpolation='bilinear')
plt.axis('off')
plt.title("📺 Word Cloud - TV Shows Descriptions", fontsize=14)

plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Word Cloud_Movies Descriptions.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Separate movies and TV shows
movies_only = df[df['type'] == 'Movie'].copy()
tv_only = df[df['type'] == 'TV Show'].copy()

# Extract numeric duration for movies (e.g., "90 min" → 90)
movies_only['duration_int'] = movies_only['duration'].str.extract('(\d+)').astype(float)

# Extract numeric seasons for TV shows (e.g., "2 Seasons" → 2)
tv_only['duration_int'] = tv_only['duration'].str.extract('(\d+)').astype(float)

# Plot Movie durations
plt.figure(figsize=(12, 5))
sns.histplot(data=movies_only, x='duration_int', bins=30, color='orangered')
plt.title("⏱️ Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Distribution of Movie Durations.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Plot TV show seasons
plt.figure(figsize=(10, 5))
sns.countplot(data=tv_only, x='duration_int', color='mediumpurple')
plt.title("📺 Number of Seasons in TV Shows")
plt.xlabel("Number of Seasons")
plt.ylabel("Count")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Number of Seasons in TV Shows.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Set plot style
sns.set(style="whitegrid")

# Remove rows with missing 'date_added'
df_date = df.dropna(subset=['date_added'])

# Extract year from 'date_added'
df_date['year_added'] = df_date['date_added'].dt.year

# Count number of contents added per year
content_per_year = df_date['year_added'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.barplot(x=content_per_year.index, y=content_per_year.values, color='coral')
plt.title('Number of Netflix Titles Added per Year', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Number of Netflix Titles Added per Year.png",
            dpi=300, bbox_inches='tight')
plt.show()

content_cumulative = content_per_year.cumsum()

plt.figure(figsize=(12, 6))
sns.lineplot(x=content_cumulative.index, y=content_cumulative.values, marker='o', color='mediumseagreen')
plt.title('Cumulative Growth of Netflix Content Over the Years', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Cumulative Titles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Cumulative Growth of Netflix Content Over the Years.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Select numerical columns only
numeric_cols = df.select_dtypes(include='number')

# Check if any numerical columns exist
if not numeric_cols.empty:
    # Compute correlation matrix
    corr_matrix = numeric_cols.corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Correlation Heatmap of Numerical Features.png",
            dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No numerical columns available for correlation heatmap.")

# Drop NaNs in 'listed_in' column
genres = df['listed_in'].dropna()

# Split by comma and flatten list
split_genres = [genre.strip() for sublist in genres.str.split(',') for genre in sublist]

# Count most common genres
genre_counts = Counter(split_genres).most_common(10)

# Unzip for plotting
top_genres, top_counts = zip(*genre_counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(top_genres), y=list(top_counts), palette='viridis')
plt.title("Top 10 Most Common Genre Labels")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"E:\Projects\Data_Analytics\Self\Netflix_EDA_Project\plots\Top 10 Most Common Genre Labels.png",
            dpi=300, bbox_inches='tight')
plt.show()


# Save cleaned file for EDA steps
df.to_csv("data/netflix_cleaned.csv", index=False)



