<h1>Word2HyperVec: From Word Embeddings to Hypervectors for Hyperdimensional Computing</h1>

<p>This repository provides a complete pipeline for classifying airline tweets as either positive or negative by converting traditional word embeddings into <strong>Hyperdimensional Computing (HDC)</strong> hypervectors. The system leverages HDC’s robustness with a novel integration of positional encoding and hypervector representation for sentiment classification tasks. Word2HyperVec is a proposed algorithm in: https://dl.acm.org/doi/10.1145/3649476.3658795</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#file-explanations">File Explanations</a></li>
  <li><a href="#workflow-summary">Workflow Summary</a></li>
  <li><a href="#methodology">Methodology</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#example-output">Example Output</a></li>
</ul>

<h2 id="overview">Overview</h2>

<p>This project combines traditional text processing, Word2Vec embeddings, and custom HDC hypervectors. It includes multiple stages:</p>
<ol>
  <li><strong>Dataset Preparation</strong>: Cleaning and splitting the dataset.</li>
  <li><strong>Positional Encoding</strong>: Adding positional information to word embeddings.</li>
  <li><strong>Hypervector Dictionary Creation</strong>: Generating high-dimensional vectors for encoding numerical data.</li>
  <li><strong>Model Training and Evaluation</strong>: Training an HDC model for sentiment classification and evaluating its performance.</li>
</ol>

<h2 id="project-structure">Project Structure</h2>

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>airline_sentiment_dataset_creation.py</code></td>
    <td>Prepares, cleans, and splits the raw tweet dataset.</td>
  </tr>
  <tr>
    <td><code>airline_pos_encoded_dataset_creation.py</code></td>
    <td>Applies positional encoding to tweet embeddings.</td>
  </tr>
  <tr>
    <td><code>dict_creation.py</code></td>
    <td>Generates a hypervector dictionary for numerical encoding.</td>
  </tr>
  <tr>
    <td><code>HDC_epoch_training_with_lr.py</code></td>
    <td>Trains the HDC model with multiple epochs and evaluates performance on test data.</td>
  </tr>
  <tr>
    <td><code>Tweets.csv</code></td>
    <td>Dataset of airline tweets with sentiment labels.</td>
  </tr>
  <tr>
    <td><code>twitter_us_airlines_word2vec_64.model</code></td>
    <td>Pre-trained Word2Vec model for embedding words in tweets.</td>
  </tr>
  <tr>
    <td><code>airline_train_data_encoded.pkl</code>, <code>airline_test_data_encoded.pkl</code></td>
    <td>Encoded positional data for model training and testing.</td>
  </tr>
</table>

<h2 id="installation">Installation</h2>

<ol>
  <li><strong>Clone the repository</strong>:
    <pre><code>git clone https://github.com/username/repo_name.git
cd repo_name</code></pre>
  </li>
  <li><strong>Install dependencies</strong>:
    <pre><code>pip install numpy pandas gensim scikit-learn keras</code></pre>
  </li>
  <li><strong>Download or add the necessary files</strong>:
    <p>Ensure <code>Tweets.csv</code> (raw tweet dataset) and <code>twitter_us_airlines_word2vec_64.model</code> (Word2Vec model) are in the root directory.</p>
  </li>
</ol>

<h2 id="usage">Usage</h2>

<p>Run each stage of the pipeline in the following order:</p>

<h3>Step 1: Data Preparation</h3>
<pre><code>python airline_sentiment_dataset_creation.py</code></pre>

<h3>Step 2: Positional Encoding</h3>
<pre><code>python airline_pos_encoded_dataset_creation.py</code></pre>

<h3>Step 3: Hypervector Dictionary Creation</h3>
<pre><code>python dict_creation.py</code></pre>

<h3>Step 4: Model Training and Evaluation</h3>
<pre><code>python HDC_epoch_training_with_lr.py</code></pre>

<h2 id="file-explanations">File Explanations</h2>

<h3>1. <code>airline_sentiment_dataset_creation.py</code></h3>
<p>This script prepares the dataset for sentiment analysis by:</p>
<ul>
  <li>Loading and filtering tweets from <code>Tweets.csv</code> to retain only those labeled as positive or negative.</li>
  <li>Cleaning each tweet (removing mentions, URLs, punctuation, and numbers).</li>
  <li>Mapping sentiments to numerical values (0 for negative, 1 for positive).</li>
  <li>Splitting the data into training and test sets (80/20 split).</li>
</ul>
<p><strong>Outputs</strong>:</p>
<ul>
  <li><code>airline_train_data.pkl</code>: Training data (tweets and labels).</li>
  <li><code>airline_test_data.pkl</code>: Testing data (tweets and labels).</li>
</ul>
<p><strong>Purpose</strong>: Ensures that the data is preprocessed and ready for encoding and model training.</p>

<h3>2. <code>airline_pos_encoded_dataset_creation.py</code></h3>
<p>This script performs positional encoding on the cleaned tweets:</p>
<ul>
  <li>Loads <code>airline_train_data.pkl</code> and <code>airline_test_data.pkl</code>.</li>
  <li>Retrieves word embeddings using a pre-trained Word2Vec model (<code>twitter_us_airlines_word2vec_64.model</code>).</li>
  <li>Applies a positional encoding method to capture the sequence structure in the embeddings.</li>
  <li>Normalizes embeddings for consistent scaling.</li>
</ul>
<p><strong>Outputs</strong>:</p>
<ul>
  <li><code>imdb_train_pos_encoded.pkl</code>: Positionally encoded training data.</li>
  <li><code>imdb_test_pos_encoded.pkl</code>: Positionally encoded test data.</li>
</ul>
<p><strong>Purpose</strong>: Adds positional information to the word embeddings, which helps the HDC model distinguish words based on their sequence in a tweet.</p>

<h3>3. <code>dict_creation.py</code></h3>
<p>This script generates a dictionary of hypervectors for encoding numerical values:</p>
<ul>
  <li>Defines a range of values with a precision step (e.g., 0.0001) and assigns each value a unique high-dimensional vector.</li>
  <li>The hypervectors are randomly initialized and flipped for each subsequent value.</li>
</ul>
<p><strong>Outputs</strong>:</p>
<ul>
  <li><code>hdc_10k_0.0001.json</code>: A JSON file mapping values to their corresponding hypervectors.</li>
</ul>
<p><strong>Purpose</strong>: Provides a mapping of numerical values to high-dimensional vectors, enabling the encoding of continuous data in HDC.</p>

<h3>4. <code>HDC_epoch_training_with_lr.py</code></h3>
<p>This script trains and evaluates the HDC model:</p>
<ul>
  <li>Loads encoded data (<code>airline_train_data_encoded.pkl</code> and <code>airline_test_data_encoded.pkl</code>) and the hypervector dictionary (<code>hdc_10k_0.0001.json</code>).</li>
  <li>Aggregates hypervectors for each sentiment (positive and negative) based on tweet embeddings.</li>
  <li>Updates hypervectors over multiple epochs and adjusts learning rates to optimize performance.</li>
  <li>After training, evaluates the model on the test data and outputs the accuracy.</li>
</ul>
<p><strong>Outputs</strong>:</p>
<ul>
  <li>Final training accuracy and time metrics printed to the console.</li>
</ul>
<p><strong>Purpose</strong>: Trains an HDC model to classify tweet sentiment based on aggregate hypervectors, providing insights into HDC’s efficacy in sentiment analysis.</p>

<h2 id="workflow-summary">Workflow Summary</h2>

<ol>
  <li><strong>Data Preprocessing</strong>:
    <ul>
      <li>Clean and split raw data in <code>Tweets.csv</code>.</li>
      <li>Outputs training and test sets in <code>airline_train_data.pkl</code> and <code>airline_test_data.pkl</code>.</li>
    </ul>
  </li>
  <li><strong>Positional Encoding</strong>:
    <ul>
      <li>Convert tweets into word embeddings using <code>twitter_us_airlines_word2vec_64.model</code>.</li>
      <li>Apply positional encoding and normalization, outputting <code>imdb_train_pos_encoded.pkl</code> and <code>imdb_test_pos_encoded.pkl</code>.</li>
    </ul>
  </li>
  <li><strong>Hypervector Dictionary Creation</strong>:
    <ul>
      <li>Generate a dictionary of hypervectors to map numerical values to high-dimensional vectors.</li>
      <li>Outputs the dictionary in <code>hdc_10k_0.0001.json</code>.</li>
    </ul>
  </li>
  <li><strong>Training and Evaluation</strong>:
    <ul>
      <li>Train the HDC model on the positionally encoded training data.</li>
      <li>Evaluate model accuracy on test data using the main code <code>HDC_epoch_training_with_lr.py</code>.</li>
    </ul>
  </li>
</ol>

