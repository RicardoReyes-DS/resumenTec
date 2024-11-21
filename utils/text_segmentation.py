# utils/text_segmentation.py

import spacy
import nltk
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Optional
from joblib import Parallel, delayed
from nltk.corpus import stopwords
import subprocess
import sys
import numpy as np

# Load SpaCy spanish model
try:
    nlp_es = spacy.load("es_core_news_sm")
except OSError:
    print("Model 'es_core_news_sm' not found. Downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp_es = spacy.load("es_core_news_sm")

# Download necessary NLTK data files once when the module is imported
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(sentences: List[str], language: str = "spanish") -> List[str]:
    """
    Preprocesses sentences by removing stopwords and non-alphanumeric characters.

    Args:
        sentences (List[str]): List of sentences to preprocess
        language (str): The language of the input text. 
                        Determines stopwords (dafault: "spanish")

    Returns:
        List[str]: Preprocesed sentences. 
    """
    stop_words = set(stopwords.words(language if language in stopwords.fileids() else "english"))
    preprocessed_sentences = []

    for sentence in sentences:
        # Remove non-alphabetic characters (except spaces)
        sentence = re.sub(r'^a-zA-Z0-9\s', '', sentence)
        # Tokenize sentence, remove stopwords
        words = [word.lower() for word in sentence.split() if word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(words))
        # Consider lematization for future sprint
    return preprocessed_sentences

def determine_n_components(X, variance_threshold: float = 0.95) -> int:
    """
    Calculate the minimum number of SVD components to meet a variance threshold

    Args:
        X: TF-IDF matrix of the text. 
        variance_threshold (float): Variance threshold for SVD. Default is 0.95.
    
    Returns:
        int: Number of components to reach desired variance.
    """
    svd = TruncatedSVD(n_components=min(X.shape) - 1)
    svd.fit(X)
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    return np.searchsorted(cumulative_variance, variance_threshold) + 1


def segment_text(
    text: str,
    num_clusters: Optional[int] = None,
    max_features: int = 1000,
    language: Optional[str] = 'spanish',
    min_clusters: int = 2,
    max_clusters: int = 10,
    svd_variance: float = 0.95,
) -> List[str]:
    """
    Segments input text into thematic chunks using KMeans clustering, 
    with a default focus on Spanish.
    
    Args:
        text (str): Input text to be segmented.
        num_clusters (Optional[int]): Number of clusters to segment the text into.
                                      If not provided the optimal number is determined automatically.
        max_features (int): Maximum number of features for the TF-IDF vectorizer. Defaults to 1000.
        language (Optional[str]): Language of the text, defaults to 'spanish'
        min_clusters: Minimum number of clusters to evaluate when determining the optimal number. 
                      Defualts to 2.
        max_clusters: Maximun number of clusters to evaluate. Defualts to 10
        svd_variance (float): Variance to retain in SVD dimensionality reduction. Defaults to 0.95.
    
    Returns:
        List[str]: A list of text chunks where each chunk represents sentences from one cluster.

    Raises:
        ValueError: If input text is empty or if invalid number of clusters is provided.
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    # Language detection logging
    logging.info(f"Using specified language: {language}")
    
    try:
        # Segment text into sentences
        if language == "spanish":
            doc = nlp_es(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = nltk.sent_tokenize(text, language="english")

        if not sentences:
            raise ValueError("No sentences found in the input text.")
        
        # Preprocess sentences (remove stopwords)
        sentences = preprocess_text(sentences, language)
        logging.info(f"Preprocessed {len(sentences)} sentences for language: {language}.")

        if not sentences:
            raise ValueError("No sentences found in the input text.")
        
        # Define custom stopwords list if Spanish
        # TF-IDF is used to convert sentences into numerical representation (feature vectors)
        if language == "spanish":
            spanish_stopwords = stopwords.words("spanish")
            vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=max_features)
        else:
            vectorizer = TfidfVectorizer(stop_words="english" if language == "english" else None, max_features=max_features)

        # Vectorize the sentences using TF-IDF
        X = vectorizer.fit_transform(sentences)

        # Reduce dimensionality using SVD
        n_components = determine_n_components(X, svd_variance)
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)
        logging.info(f"Dimensionality reduced to {X_reduced.shape[1]} components.")

        # KMeans clustering
        # Determine the optimal number of clusters if not provided
        if num_clusters is None:
            max_possible_clusters = min(max_clusters, len(sentences) - 1)
            if max_possible_clusters < min_clusters:
                num_clusters = 1
            else:
                def evaluate_k(k):
                    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
                    labels = kmeans.fit_predict(X_reduced)
                    score = silhouette_score(X_reduced, labels)
                    logging.info(f"Silhouette Score for k={k}: {score}")
                    return (k,score)

                scores = Parallel(n_jobs=-1)(delayed(evaluate_k)(k) for k in range(min_clusters, max_possible_clusters + 1))
                num_clusters = max(scores, key=lambda item: item[1])[0]
                logging.info(f"Optimal number of clusters determined: {num_clusters}")
        
        # Validate number of clusters
        if num_clusters <= 0 or num_clusters > len(sentences):
            raise ValueError("Invalid number of clusters")

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(X_reduced)

        # Group sentences by cluster labels
        clusters = {label: [] for label in set(labels)}
        for sentence, label in zip(sentences, labels):
            clusters[label].append(sentence)

        # Combine sentences in each cluster to form chunks
        chunks = [' '.join(clusters[label]) for label in sorted(clusters.keys())]

        logging.info(f"Text segmentation completed succesfully into {num_clusters} clusters.")
        return chunks
    
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        raise

    except Exception as e:
        logging.error(f"Failed to segment text: {e}")
        raise