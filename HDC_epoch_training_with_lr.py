import numpy as np
import json
import pickle
import time
from gensim.models import Word2Vec

# Assuming cosine_similarity_1d is defined as before
def cosine_similarity_1d(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0 if magnitude == 0 else dot_product / magnitude

def predict_sentiment_optimized(test_embeddings, negative_vector, positive_vector, hypervectors):
    pred_list = []
    for embedding in test_embeddings:
        keys = [f"{np.round(val, 3):.3f}" for val in np.nditer(embedding)]
        keys = [key if key != "-0.000" else "0.000" for key in keys]
        valid_hypervectors = [hypervectors[key] for key in keys if key in hypervectors]

        if valid_hypervectors:
            pos_hypervector = sum(valid_hypervectors)
            negative_sim = cosine_similarity_1d(pos_hypervector, negative_vector)
            positive_sim = cosine_similarity_1d(pos_hypervector, positive_vector)
            pred_list.append(0 if negative_sim > positive_sim else 1)
    return pred_list

def calculate_hypervectors(X, hypervectors):
    precomputed_hypervectors = []
    for embedding in X:
        keys = [f"{np.round(val, 3):.3f}" for val in embedding]
        keys = [key if key != "-0.000" else "0.000" for key in keys]
        valid_hypervectors = [hypervectors[key] for key in keys if key in hypervectors]
        if valid_hypervectors:
            pos_hypervector = sum(valid_hypervectors)
            # pos_hypervector = np.where(pos_hypervector >= 1, 1, -1)
        else:
            pos_hypervector = np.zeros_like(next(iter(hypervectors.values())))  # Assuming all hypervectors have the same shape
        precomputed_hypervectors.append(pos_hypervector)
    return precomputed_hypervectors

def predict_with_precomputed_hypervectors(precomputed_hypervectors, negative_vector, positive_vector):
    pred_list = []
    for pos_hypervector in precomputed_hypervectors:
        negative_sim = cosine_similarity_1d(pos_hypervector, negative_vector)
        positive_sim = cosine_similarity_1d(pos_hypervector, positive_vector)
        pred_list.append(0 if negative_sim > positive_sim else 1)
    return pred_list

def train_and_predict(X_train, y_train, hypervectors, epochs=50):
    
    negative_vector, positive_vector = np.zeros(1000), np.zeros(1000)
    update_indices = np.arange(len(X_train))
    time_start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} starts")
        
        for index in update_indices:
            label = y_train[index]
            embedding = X_train[index]
            target_aggregated_embd = negative_vector if label == 0 else positive_vector
            
        
            index_key_strs = [f"{np.round(val, 3):.3f}" for val in embedding]
            # index_key_strs = [f"{val:.3f}" for val in rounded_vector]
            index_key_strs = [key if key != "-0.000" else "0.000" for key in index_key_strs]
            
            for key in index_key_strs:
                hypervector_value = hypervectors.get(key, None)
                if hypervector_value is not None:
                    if epoch == 0:
                        target_aggregated_embd += hypervector_value
                    else:
                        target_aggregated_embd += np.array(hypervector_value) * (1 + 1)
        
        negative_vector_bin = np.where(negative_vector >= 1, 1, -1)
        positive_vector_bin = np.where(positive_vector >= 1, 1, -1)

        if epoch !=  epochs - 1:
            predictions = predict_with_precomputed_hypervectors(precomputed_hypervectors_train, negative_vector, positive_vector)
            correct_predictions = [pred == actual for pred, actual in zip(predictions, y_train)]
            accuracy = sum(correct_predictions) / len(y_train) * 100
            print(f"Epoch {epoch+1}: Accuracy = {accuracy}% Cosine Similarity of classification vectors{cosine_similarity_1d(negative_vector_bin, positive_vector_bin)}")
        else: 
            predictions = predict_with_precomputed_hypervectors(precomputed_hypervectors_train, negative_vector, positive_vector)
            time_end = time.time()
            correct_predictions = [pred == actual for pred, actual in zip(predictions, y_train)]
            accuracy = sum(correct_predictions) / len(y_train) * 100
            print(f"Epoch {epoch+1}: Accuracy = {accuracy}%")

        if epoch < epochs - 1:
            correct_indices = [i for i, correct in enumerate(correct_predictions) if correct]
            update_indices = correct_indices

    
    print("Total time for training and testing: ", time_end - time_start)
    print(f"Final Accuracy: {accuracy}%, Cosine sim of classification vecs: {cosine_similarity_1d(negative_vector, positive_vector)}")
    return negative_vector, positive_vector


# Load data and hypervector dictionary
dict_path = "dict//hdc_1k_0.001.json"
training_embeddings_file = "data//airline_train_data_encoded.pkl"
testing_embeddings_file = "data//airline_test_data_encoded.pkl"
model_path = "ml_models//lr_model_v2.joblib"
from gensim.models import Word2Vec

# Function to load data from pickle files
def load_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels

# Function to convert tweets to average word vectors
def tweet_to_avg_vector(tweet, model):
    vector = np.zeros(model.vector_size)
    num_words = 0
    for word in tweet:
        if word in model.wv:
            vector += model.wv[word]
            num_words += 1
    if num_words > 0:
        vector = vector
    return vector

# Load the saved Word2Vec model trained with airline data
word2vec_model_path = "twitter_us_airlines_word2vec_64.model"  # Update this path as necessary
word2vec_model = Word2Vec.load(word2vec_model_path)

with open(dict_path, 'r') as file:
    hypervector_dict = json.load(file)
    hypervector_dict = {k: np.array(v) for k, v in hypervector_dict.items()}

import joblib
model = joblib.load(model_path)

weights = model.coef_[0]
bias = model.intercept_[0]
bias_per_feature = bias / len(weights)

# Load training and testing data
train_data_path = "data//airline_train_data.pkl"  # Update this path as necessary
test_data_path = "data//airline_test_data.pkl"  # Update this path as necessary

X_train_raw, y_train = load_data(train_data_path)
X_test_raw, y_test = load_data(test_data_path)

# Convert tweets in training and testing sets to average word vectors
X_train = [tweet_to_avg_vector(tweet, word2vec_model) for tweet in X_train_raw]
X_test = [tweet_to_avg_vector(tweet, word2vec_model) for tweet in X_test_raw]

# from sklearn.preprocessing import StandardScaler
# scaler_weights = StandardScaler()
# weights_standardized = scaler_weights.fit_transform(weights.reshape(-1, 1)).flatten()


weighted_X_train = X_train * weights 
weighted_X_test = X_test * weights 


from sklearn.preprocessing import MinMaxScaler
# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler to the weighted training data and transform it
weighted_X_train_normalized = scaler.fit_transform(weighted_X_train)
# Transform the weighted testing data based on the scaler fitted to the training data
weighted_X_test_normalized = scaler.transform(weighted_X_test)


# with open('test_100_samples.pkl', 'wb') as f:
#     pickle.dump([weighted_X_test_normalized[0:100]], f)

# Execute training and prediction
precomputed_hypervectors_train = calculate_hypervectors(weighted_X_train_normalized, hypervector_dict)
precomputed_hypervectors_test = calculate_hypervectors(weighted_X_test_normalized, hypervector_dict)

negative_vector, positive_vector = train_and_predict(weighted_X_train_normalized, y_train, hypervector_dict, epochs=50)

# # Save the vectors to a binary file
# with open('hdc_final_version.bin', 'wb') as f:
#     pickle.dump({"negative": negative_vector, "positive": positive_vector}, f)


test_preds = predict_with_precomputed_hypervectors(precomputed_hypervectors_test, negative_vector, positive_vector)
correct_predictions = [pred == actual for pred, actual in zip(test_preds, y_test)]
accuracy = sum(correct_predictions) / len(y_test) * 100
print(f"Final Test Accuracy = {accuracy}%")
