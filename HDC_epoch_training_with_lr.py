import numpy as np
import json
import pickle
import time

def aggregate_embeddings(training_embeddings, training_labels, hypervectors, lr, binarize=False):
    bot_aggregated_embd = np.zeros(1000)
    normal_aggregated_emd = np.zeros(1000)
    
    for label, embedding in zip(training_labels, training_embeddings):
        target_aggregated_embd = bot_aggregated_embd if label == 0 else normal_aggregated_emd
        for vector in embedding:
            rounded_vector = np.round(vector, decimals=3)
            index_key_strs = [f"{val:.3f}" for val in rounded_vector]
            index_key_strs = [key if key != "-0.000" else "0.000" for key in index_key_strs]
            for key in index_key_strs:
                hypervector_value = hypervectors.get(key, None)
                if hypervector_value is not None:
                    # Scale the contribution by (1 + lr)
                    target_aggregated_embd += np.array(hypervector_value) * (1 + lr)
    
    if binarize:
        bot_aggregated_embd = np.where(bot_aggregated_embd >= 1, 1, -1)
        normal_aggregated_emd = np.where(normal_aggregated_emd >= 1, 1, -1)

    return bot_aggregated_embd, normal_aggregated_emd

def predict_sentiment_optimized(test_embeddings, bot_vector, normal_vector, hypervectors):
    pred_list = []
    for embedding in test_embeddings:
        keys = [f"{np.round(val, 3):.3f}" for val in np.nditer(embedding)]
        keys = [key if key != "-0.000" else "0.000" for key in keys]
        valid_hypervectors = [hypervectors[key] for key in keys if key in hypervectors]
        
        if valid_hypervectors:
            pos_hypervector = sum(valid_hypervectors)
            pos_hypervector = np.where(pos_hypervector >= 1, 1, -1)
            bot_sim = cosine_similarity_1d(pos_hypervector, bot_vector)
            normal_sim = cosine_similarity_1d(pos_hypervector, normal_vector)
            pred_list.append(0 if bot_sim > normal_sim else 1)
    return pred_list

def cosine_similarity_1d(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude if magnitude else 0

# Training and Prediction Logic
def train_and_predict(X_train, y_train, X_test, y_test, hypervector_dict, epochs=2, lr=1):
    time_start = time.time()
    
    for epoch in range(epochs):
        binarize = (epoch >= 1)  # Binarize only after the second epoch
        bot_vector, normal_vector = aggregate_embeddings(X_train, y_train, hypervector_dict, lr, binarize)
        print(f"Epoch {epoch+1} aggregation complete.")
        
        if epoch == epochs - 1:  # Perform predictions after the last epoch
            predictions = predict_sentiment_optimized(X_test, bot_vector, normal_vector, hypervector_dict)
            correct_predictions = sum(pred == actual for pred, actual in zip(predictions, y_test))
            accuracy = correct_predictions / len(X_test) * 100
            print(f"Final Epoch {epoch+1}: Accuracy = {accuracy}%")
    
    time_end = time.time()
    print("Total training and testing time: ", time_end - time_start)
    print(f"Final Accuracy: {accuracy}%, Cosine sim of classification vecs: {cosine_similarity_1d(bot_vector, normal_vector)}")

# Load Data and Hypervector Dictionary
dict_path = "dict//hdc_1k_0.001.json"
training_embeddings_file = "data//airline_train_data_encoded.pkl"
testing_embeddings_file = "data//airline_test_data_encoded.pkl"

with open(dict_path, 'r') as file:
    hypervector_dict = json.load(file)
    hypervector_dict = {key: np.array(value) for key, value in hypervector_dict.items()}

with open(training_embeddings_file, 'rb') as f:
    X_train, y_train = pickle.load(f)

with open(testing_embeddings_file, 'rb') as f:
    X_test, y_test = pickle.load(f)

# Train and Predict
train_and_predict(X_train, y_train, X_test, y_test, hypervector_dict, epochs=2, lr=0.1)
