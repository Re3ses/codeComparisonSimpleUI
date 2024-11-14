# functions.py
import re
import os
import numpy as np
import json
from tqdm import tqdm
from tree_sitter import Language, Parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from transformers import RobertaTokenizer, RobertaModel
from typing import List, Dict, Tuple
import ast
from difflib import SequenceMatcher

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
    elif language == 'cpp':
        code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove string literals
    code = re.sub(r'".*?"', '""', code)

    # Remove import statements
    if language == 'java':
        code = re.sub(r'import\s+[\w.]+;', '', code)
    elif language == 'python':
        code = re.sub(r'import\s+[\w.]+|from\s+[\w.]+\s+import\s+[\w.]+', '', code)
    elif language == 'cpp':
        code = re.sub(r'#include\s+<[\w.]+>|#include\s+"[\w.]+"', '', code)

    # Remove package declarations (Java only)
    if language == 'java':
        code = re.sub(r'package\s+[\w.]+;', '', code)

    # Remove whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def jaccard_similarity(tokens1, tokens2):
    """Compute Jaccard similarity between two token sets"""
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def normalized_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
def get_codebert_embedding(code):
    try:
        if tokenizer_type == 'default':
            inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif tokenizer_type == 'word':
            tokens = code.split()
            inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif tokenizer_type == 'character':
            tokens = list(code)
            inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error generating CodeBERT embedding: {str(e)}")
        return None
def get_structural_similarity(code1: str, code2: str) -> float:
    """Compare code structure using AST"""
    try:
        # Normalize both codes
        norm_code1 = self.preprocess_code(code1)
        norm_code2 = self.preprocess_code(code2)
        
        # Use sequence matcher for structural comparison
        return SequenceMatcher(None, norm_code1, norm_code2).ratio()
    except:
        return 0.0

def get_token_similarity(code1: str, code2: str) -> float:
    """Compare code based on token sequences"""
    def get_tokens(code):
        try:
            return [token.string for token in ast.walk(ast.parse(code))]
        except:
            return code.split()
            
    tokens1 = set(get_tokens(code1))
    tokens2 = set(get_tokens(code2))
    
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)

def compute_similarities(submissions: dict) -> tuple:
    """
    Compute similarity matrices for a set of code submissions using multiple metrics:
    structural (AST), token-based, semantic (CodeBERT), Jaccard, and TF-IDF similarities.
    
    Args:
        submissions (dict): A dictionary where keys are filenames and values are dictionaries containing:
            - 'sequence' (str): The code sequence
            - 'embedding' (np.ndarray): The CodeBERT embedding of the code sequence
            - 'tokens' (list of str): The tokens of the code sequence
            - 'ast' (Any): The AST representation of the code
            
    Returns:
        tuple: A tuple containing five numpy arrays:
            - structural_similarities (np.ndarray): A 2D array of AST-based structural similarity scores
            - token_similarities (np.ndarray): A 2D array of token-based similarity scores
            - semantic_similarities (np.ndarray): A 2D array of CodeBERT semantic similarity scores
            - jaccard_similarities (np.ndarray): A 2D array of Jaccard similarity scores
            - tfidf_similarities (np.ndarray): A 2D array of TF-IDF similarity scores
    """
    filenames = list(submissions.keys())
    n = len(filenames)
    
    # Initialize similarity matrices
    structural_similarities = np.zeros((n, n))
    token_similarities = np.zeros((n, n))
    semantic_similarities = np.zeros((n, n))
    jaccard_similarities = np.zeros((n, n))
    tfidf_similarities = np.zeros((n, n))

    # Prepare TF-IDF vectorizer for structural similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sub['sequence'] for sub in submissions.values()])

    # Get all embeddings for vectorized computation
    embeddings = np.array([sub['embedding'] for sub in submissions.values()])

    # Compute similarities for all pairs
    for i in range(n):
        for j in range(i+1, n):
            file1, file2 = filenames[i], filenames[j]
            sub1, sub2 = submissions[file1], submissions[file2]
            
            # Structural similarity (AST-based)
            structural_sim = get_structural_similarity(sub1['ast'], sub2['ast'])
            structural_similarities[i][j] = structural_similarities[j][i] = structural_sim * 100
            
            # Token-based similarity
            token_sim = get_token_similarity(sub1['tokens'], sub2['tokens'])
            token_similarities[i][j] = token_similarities[j][i] = token_sim * 100
            
            # Semantic similarity (CodeBERT)
            semantic_sim = normalized_similarity(embeddings[i], embeddings[j])
            semantic_similarities[i][j] = semantic_similarities[j][i] = semantic_sim * 100
            
            # Jaccard similarity
            jaccard_sim = jaccard_similarity(sub1['tokens'], sub2['tokens'])
            jaccard_similarities[i][j] = jaccard_similarities[j][i] = jaccard_sim * 100
            
            # TF-IDF similarity
            tfidf_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            tfidf_similarities[i][j] = tfidf_similarities[j][i] = tfidf_sim * 100
        
        # Set diagonal to 100% similarity
        structural_similarities[i][i] = token_similarities[i][i] = 100
        semantic_similarities[i][i] = jaccard_similarities[i][i] = 100
        tfidf_similarities[i][i] = 100

    return (
        structural_similarities,
        token_similarities,
        semantic_similarities,
        jaccard_similarities,
        tfidf_similarities
    )

