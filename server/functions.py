import re
import os
import json
import numpy as np
from tqdm import tqdm
from tree_sitter import Language, Parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import RobertaTokenizer, RobertaModel

# Load Java, Python, and C++ languages
import tree_sitter_java
import tree_sitter_python
import tree_sitter_cpp

JAVA_LANGUAGE = Language(tree_sitter_java.language())
PYTHON_LANGUAGE = Language(tree_sitter_python.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())

java_parser = Parser(JAVA_LANGUAGE)
python_parser = Parser(PYTHON_LANGUAGE)
cpp_parser = Parser(CPP_LANGUAGE)

# Load CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    
def tree_to_sequence(code, language):
    if language == 'java':
        parser = java_parser
    elif language == 'python':
        parser = python_parser
    elif language == 'cpp':
        parser = cpp_parser
    else:
        raise ValueError("Unsupported language")

    tree = parser.parse(bytes(code, "utf8"))

    def traverse(node, depth=0):
        if node.type != 'string' and node.type != 'comment':
            yield f"{node.type}_{depth}"
            for child in node.children:
                yield from traverse(child, depth + 1)

    return ' '.join(traverse(tree.root_node))

def preprocess_code(code, language):
    # Remove comments
    if language == 'java':
        code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
    elif language == 'python':
        code = re.sub(r'#.*?\n|\'\'\'.*?\'\'\'|""".*?"""', '', code, flags=re.DOTALL)

    # Remove string literals
    code = re.sub(r'".*?"', '""', code)

    # Remove import statements
    if language == 'java':
        code = re.sub(r'import\s+[\w.]+;', '', code)
    elif language == 'python':
        code = re.sub(r'import\s+[\w.]+|from\s+[\w.]+\s+import\s+[\w.]+', '', code)

    # Remove package declarations (Java only)
    if language == 'java':
        code = re.sub(r'package\s+[\w.]+;', '', code)

    # Remove whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    return code


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def normalized_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
def get_codebert_embedding(code):
    try:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error generating CodeBERT embedding: {str(e)}")
        return None
    

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