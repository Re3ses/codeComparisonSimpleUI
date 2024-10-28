import os
import numpy as np
from tqdm import tqdm
from tree_sitter import Language, Parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import RobertaTokenizer, RobertaModel

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def compute_similarities(submissions):
    """
    Compute similarity matrices for a set of code submissions using CodeBERT embeddings, Jaccard similarity, and TF-IDF similarity.
    Args:
        submissions (dict): A dictionary where keys are filenames and values are dictionaries containing:
            - 'sequence' (str): The code sequence.
            - 'embedding' (np.ndarray): The CodeBERT embedding of the code sequence.
            - 'tokens' (list of str): The tokens of the code sequence.
    Returns:
        tuple: A tuple containing three numpy arrays:
            - codebert_similarities (np.ndarray): A 2D array of CodeBERT similarity scores between submissions.
            - jaccard_similarities (np.ndarray): A 2D array of Jaccard similarity scores between submissions.
            - tfidf_similarities (np.ndarray): A 2D array of TF-IDF similarity scores between submissions.
    """
    filenames = list(submissions.keys())
    n = len(filenames)
    codebert_similarities = np.zeros((n, n))
    jaccard_similarities = np.zeros((n, n))
    tfidf_similarities = np.zeros((n, n))

    # Prepare TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sub['sequence'] for sub in submissions.values()])

    embeddings = np.array([sub['embedding'] for sub in submissions.values()])

    for i in range(n):
        for j in range(i+1, n):
            # CodeBERT similarity
            codebert_sim = normalized_similarity(embeddings[i], embeddings[j])
            codebert_similarities[i][j] = codebert_similarities[j][i] = codebert_sim * 100

            # Jaccard similarity
            jaccard_sim = jaccard_similarity(submissions[filenames[i]]['tokens'], submissions[filenames[j]]['tokens'])
            jaccard_similarities[i][j] = jaccard_similarities[j][i] = jaccard_sim * 100

            # TF-IDF similarity
            tfidf_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            tfidf_similarities[i][j] = tfidf_similarities[j][i] = tfidf_sim * 100

    return codebert_similarities, jaccard_similarities, tfidf_similarities

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def normalized_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
