import numpy as np
import pickle
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def positional_encoding(position, d_model):
    def get_angle(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angle(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return pos_encoding

def encode_tweets_with_position(tweets, model):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    encoded_tweets = []
    for tweet in tweets:
        words = tweet.split()
        word_embeddings = np.array([model.wv[word] if word in model.wv else np.zeros(model.vector_size) for word in words])
        pos_encodings = positional_encoding(len(words), model.vector_size)
        encoded_tweet = word_embeddings + pos_encodings
        encoded_tweets.append(encoded_tweet)
    return encoded_tweets

def normalize_embeddings(embeddings):
    # Apply Min-Max normalization to each embedding individually
    normalized_embeddings = []
    for emb in embeddings:
        min_val = np.min(emb, axis=0, keepdims=True)
        max_val = np.max(emb, axis=0, keepdims=True)
        # Avoid division by zero
        range_val = np.where(max_val - min_val == 0, 1, max_val - min_val)
        normalized_emb = (emb - min_val) / range_val
        normalized_embeddings.append(normalized_emb)
    return normalized_embeddings


# Load the pre-trained Word2Vec model
word2vec_model_path = 'word2vec_imdb_64.model'
model = Word2Vec.load(word2vec_model_path)

# Load training and testing data
with open('imdb_train_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('imdb_test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Encode tweets with positional information for training and testing sets
X_train_encoded = encode_tweets_with_position(X_train, model)
X_test_encoded = encode_tweets_with_position(X_test, model)

X_train_norm = normalize_embeddings(X_train_encoded)
X_test_norm = normalize_embeddings(X_test_encoded)

# Save the processed data to new pickle files
with open('imdb_train_pos_encoded.pkl', 'wb') as f:
    pickle.dump((X_train_norm, y_train), f)

with open('imdb_test_pos_encoded.pkl', 'wb') as f:
    pickle.dump((X_test_norm, y_test), f)
