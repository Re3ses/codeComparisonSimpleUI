!pip install tree-sitter
!pip install tree-sitter-java
!pip install tree-sitter-python
!pip install tree-sitter-cpp

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

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def normalized_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_file_language(filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension in ['.java']:
        return 'java'
    elif extension in ['.py', '.pyw']:
        return 'python'
    elif extension in ['.cpp', '.cxx', '.cc', '.c++', '.hpp', '.hxx', '.hh', '.h++', '.h']:
        return 'cpp'
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def get_codebert_embedding(code, tokenizer_type='default'):
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

def process_files(directory, tokenizer_type='default', use_tree_sitter=False):
    submissions = {}
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                language = get_file_language(file)
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        code = f.read()
                        preprocessed_code = preprocess_code(code, language)
                        if use_tree_sitter:
                            tree_sequence = tree_to_sequence(preprocessed_code, language)
                            codebert_embedding = get_codebert_embedding(tree_sequence, tokenizer_type)
                            tokens = set(tree_sequence.split())
                        else:
                            codebert_embedding = get_codebert_embedding(preprocessed_code, tokenizer_type)
                            tokens = set(preprocessed_code.split())
                        submission = {
                            'sequence': tree_sequence if use_tree_sitter else preprocessed_code,
                            'language': language,
                            'embedding': codebert_embedding,
                            'tokens': tokens
                        }
                        submissions[file] = submission
                    except UnicodeDecodeError:
                        print(f"Error reading {file_path}. Skipping.")
            except ValueError as e:
                print(f"Skipping file {file}: {str(e)}")
    return submissions

def run_all_combinations(directory):
    tokenizer_options = ['default', 'word', 'character']
    tree_sitter_options = [False, True]
    all_results = {}

    for tokenizer_type in tokenizer_options:
        for use_tree_sitter in tree_sitter_options:
            print(f"Running with tokenizer: {tokenizer_type}, Tree-Sitter: {'Yes' if use_tree_sitter else 'No'}")
            plagiarism_results = check_plagiarism(directory, tokenizer_type=tokenizer_type, use_tree_sitter=use_tree_sitter)

            key = f"{tokenizer_type}_{'tree_sitter' if use_tree_sitter else 'no_tree_sitter'}"
            all_results[key] = plagiarism_results

    return all_results

def compute_similarities(submissions):
    filenames = list(submissions.keys())
    n = len(filenames)
    semantic_similarities = np.zeros((n, n))
    token_similarities = np.zeros((n, n))
    structural_similarities = np.zeros((n, n))

    # Prepare TF-IDF vectorizer for structural similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sub['sequence'] for sub in submissions.values()])

    embeddings = np.array([sub['embedding'] for sub in submissions.values()])

    for i in range(n):
        for j in range(i+1, n):
            # Semantic similarity (formerly CodeBERT)
            semantic_sim = normalized_similarity(embeddings[i], embeddings[j])
            semantic_similarities[i][j] = semantic_similarities[j][i] = semantic_sim * 100

            # Token similarity (formerly Jaccard)
            token_sim = jaccard_similarity(submissions[filenames[i]]['tokens'], submissions[filenames[j]]['tokens'])
            token_similarities[i][j] = token_similarities[j][i] = token_sim * 100

            # Structural similarity (formerly TF-IDF)
            structural_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            structural_similarities[i][j] = structural_similarities[j][i] = structural_sim * 100

    return semantic_similarities, token_similarities, structural_similarities

def check_plagiarism(directory, threshold=75, tokenizer_type='default', use_tree_sitter=False):
    submissions = process_files(directory, tokenizer_type, use_tree_sitter)
    semantic_similarities, token_similarities, structural_similarities = compute_similarities(submissions)

    filenames = list(submissions.keys())
    n = len(filenames)

    results = []
    total_similarity = 0
    comparison_count = 0

    for i in range(n):
        file_result = {"file": filenames[i], "comparisons": {}}
        for j in range(n):
            if i != j:
                semantic_sim = semantic_similarities[i][j]
                token_sim = token_similarities[i][j]
                structural_sim = structural_similarities[i][j]

                # Calculate weighted combined similarity
                combined_sim = (
                    token_sim * 0.45 +          # Token similarity weight
                    structural_sim * 0.45 +      # Structural similarity weight
                    semantic_sim * 0.05          # Semantic similarity weight
                )

                file_result["comparisons"][filenames[j]] = {
                    "token_similarity": token_sim,
                    "structural_similarity": structural_sim,
                    "semantic_similarity": semantic_sim,
                    "combined_similarity": combined_sim,
                    "potential_plagiarism": combined_sim > threshold
                }

                total_similarity += combined_sim
                comparison_count += 1

        results.append(file_result)

    average_similarity = total_similarity / comparison_count if comparison_count > 0 else 0

    return {
        "threshold": threshold,
        "weights": {
            "token_similarity": 0.45,
            "structural_similarity": 0.45,
            "semantic_similarity": 0.05
        },
        "average_similarity": average_similarity,
        "results": results
    }

# User input for analysis type
print("Choose an analysis type:")
print("1. Single configuration")
print("2. All combinations")
analysis_choice = input("Enter your choice (1/2): ")

if analysis_choice == '1':
    # Single configuration
    print("Choose a tokenizer:")
    print("1. Default")
    print("2. Word")
    print("3. Character")
    tokenizer_choice = input("Enter your choice (1/2/3): ")

    if tokenizer_choice == '1':
        tokenizer_type = 'default'
    elif tokenizer_choice == '2':
        tokenizer_type = 'word'
    elif tokenizer_choice == '3':
        tokenizer_type = 'character'
    else:
        print("Invalid choice. Using default tokenizer.")
        tokenizer_type = 'default'

    use_tree_sitter = input("Use Tree-Sitter? (y/n): ").lower() == 'y'

    # Example usage
    directory = '/kaggle/input/ir-plag-dataset'
    plagiarism_results = check_plagiarism(directory, tokenizer_type=tokenizer_type, use_tree_sitter=use_tree_sitter)

    # Save to JSON file using the custom encoder
    filename = f'plagiarism_results_{tokenizer_type}_{"tree_sitter" if use_tree_sitter else "no_tree_sitter"}.json'
    with open(filename, 'w') as f:
        json.dump(plagiarism_results, f, indent=2, cls=NumpyEncoder)

    print(f"Results have been saved to '{filename}'")
    print(f"Average similarity score: {plagiarism_results['average_similarity']:.2f}")

elif analysis_choice == '2':
    # All combinations
    directory = '/kaggle/input/ir-plag-dataset'
    all_results = run_all_combinations(directory)

    # Save all results to a single JSON file
    filename = 'plagiarism_results_all_combinations.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"Results for all combinations have been saved to '{filename}'")

    # Print average similarity scores for each combination
    print("\nAverage similarity scores:")
    for key, results in all_results.items():
        print(f"{key}: {results['average_similarity']:.2f}")

else:
    print("Invalid choice. Exiting.")