import pandas as pd
import re
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('Tweets.csv')

# Filter out only positive and negative tweets
df_filtered = df[df['airline_sentiment'].isin(['positive', 'negative'])]

# Clean the tweets: remove airline mentions
def clean_tweet(tweet):
    # Remove mentions (@something)
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove any punctuation and numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

df_filtered['text'] = df_filtered['text'].apply(clean_tweet)

# Map sentiment to numerical values: 0 for negative, 1 for positive
df_filtered['airline_sentiment'] = df_filtered['airline_sentiment'].map({'negative': 0, 'positive': 1})

# Split the data into features (X) and labels (y)
X = df_filtered['text'].values
y = df_filtered['airline_sentiment'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing data to pickle files
with open('airline_train_data.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)

with open('airline_test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

print("Data preprocessing complete and saved to pickle files.")
