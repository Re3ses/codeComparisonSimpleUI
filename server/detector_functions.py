# detector_functions.py
import re
import os
import json
import ast
from difflib import SequenceMatcher
from tree_sitter import Language, Parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import RobertaTokenizer, RobertaModel
from typing import Dict, Set, List, Any

# for performance testing
import time

# Load Java, Python, and C++ languages
import tree_sitter_java
import tree_sitter_python
import tree_sitter_cpp

JAVA_LANGUAGE = Language(tree_sitter_java.language())
PYTHON_LANGUAGE = Language(tree_sitter_python.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())

class EnhancedCodeSimilarityDetector:
    def __init__(self, model_name='microsoft/codebert-base'):
        # Initialize CodeBERT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Initialize tree-sitter parsers
        self.parsers = {
            'java': Parser(JAVA_LANGUAGE),
            'python': Parser(PYTHON_LANGUAGE),
            'cpp': Parser(CPP_LANGUAGE),
        }

    def preprocess_code(self, code: str, language: str = 'python') -> str:
        """
        Comprehensive code preprocessing combining approaches from both files
        """
        # Remove comments based on language
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

        # Normalize whitespace
        code = ' '.join(code.split())

        try:
            # Additional AST-based normalization for Python
            if language == 'python':
                tree = ast.parse(code)
                code = self.normalize_ast(tree)
        except:
            pass

        return code

    def normalize_ast(self, node, counter=None):
        """Normalize variable names in AST"""
        if counter is None:
            counter = {'var': 0}
            
        if isinstance(node, ast.Name):
            return f'var_{counter["var"]}'
        elif isinstance(node, ast.arg):
            counter['var'] += 1
            return f'var_{counter["var"]}'
        
        for child in ast.iter_child_nodes(node):
            self.normalize_ast(child, counter)
        
        return ast.unparse(node)

    def get_tree_sitter_sequence(self, code: str, language: str) -> str:
        """Get tree-sitter sequence representation of code"""
        try:
            parser = self.parsers.get(language)
            if parser:
                tree = parser.parse(bytes(code, "utf8"))
                
                def traverse(node, depth=0):
                    if node.type != 'string' and node.type != 'comment':
                        yield f"{node.type}_{depth}"
                        for child in node.children:
                            yield from traverse(child, depth + 1)

                return ' '.join(traverse(tree.root_node))
        except:
            return code

    def get_structural_similarity(self, code1: str, code2: str, language: str = 'python') -> float:
        """
        Enhanced structural similarity using both AST and tree-sitter
        """
        try:
            # Normalize both codes
            norm_code1 = self.preprocess_code(code1, language)
            norm_code2 = self.preprocess_code(code2, language)
            
            # Get tree-sitter sequences
            seq1 = self.get_tree_sitter_sequence(norm_code1, language)
            seq2 = self.get_tree_sitter_sequence(norm_code2, language)
            
            # Combine SequenceMatcher and tree-sitter similarities
            sequence_sim = SequenceMatcher(None, norm_code1, norm_code2).ratio()
            tree_sim = SequenceMatcher(None, seq1, seq2).ratio()
            
            return (sequence_sim + tree_sim) / 2
        except:
            return 0.0

    def get_token_similarity(self, code1: str, code2: str) -> float:
        """
        Enhanced token similarity using Jaccard similarity
        """
        def get_tokens(code: str) -> Set[str]:
            try:
                return {token.string for token in ast.walk(ast.parse(code))}
            except:
                return set(code.split())

        tokens1 = get_tokens(code1)
        tokens2 = get_tokens(code2)
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union != 0 else 0.0

    def get_tfidf_similarity(self, code1: str, code2: str) -> float:
        """
        TF-IDF based similarity measurement
        """
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([code1, code2])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except:
            return 0.0
        
    def get_codebert_embedding(self, code: str, tokenizer_type='default') -> Any:
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


    def get_semantic_similarity(self, code1: str, code2: str, tokenizer: str) -> float:
        """
        Semantic similarity using CodeBERT embeddings
        """
        try:
            # Tokenize codes
            tokens1 = self.get_codebert_embeddings(code1, tokenizer) 
            tokens2 = self.get_codebert_embeddings(code2, tokenizer) 
            
            # Move to device
            tokens1 = {k: v.to(self.device) for k, v in tokens1.items()}
            tokens2 = {k: v.to(self.device) for k, v in tokens2.items()}
            
            # Generate embeddings
            with torch.no_grad():
                emb1 = self.model(**tokens1).last_hidden_state[:, 0, :].cpu().numpy()
                emb2 = self.model(**tokens2).last_hidden_state[:, 0, :].cpu().numpy()
                
            return float(cosine_similarity(emb1, emb2)[0][0])
        except:
            return 0.0

    def compute_similarity(self, code1: str, code2: str, language: str = 'python', weights: Dict[str, float] = None, tokenizer: str = 'default') -> Dict:
        """
        Compute comprehensive similarity score using all metrics and custom weights
        """
        if weights is None:
            weights = {
                'structural': 0.3,  # AST and tree-sitter structure
                'token': 0.2,      # Token-based similarity
                'tfidf': 0.2,      # TF-IDF similarity
                'semantic': 0.3    # CodeBERT embeddings
            }

        start_time = time.time()
            
        # Get individual similarity scores
        structural_sim = self.get_structural_similarity(code1, code2, language)
        token_sim = self.get_token_similarity(code1, code2)
        tfidf_sim = self.get_tfidf_similarity(code1, code2)
        semantic_sim = self.get_semantic_similarity(code1, code2, tokenizer)
        
        # Calculate weighted average
        weighted_sim = (
            weights['structural'] * structural_sim +
            weights['token'] * token_sim +
            weights['tfidf'] * tfidf_sim +
            weights['semantic'] * semantic_sim
        )

        print(f"Time taken: {time.time() - start_time}")
        
        return {
            'total_similarity': weighted_sim,
            'structural_similarity': structural_sim,
            'token_similarity': token_sim,
            'tfidf_similarity': tfidf_sim,
            'semantic_similarity': semantic_sim
        }

    def process_submission(self, submission):
        """
        Process a single submission using the enhanced detector
        """
        try:
            language = submission['language']
            code = submission['code']
            
            # Preprocess the code
            preprocessed_code = self.preprocess_code(code, language)
            
            return {
                'code': preprocessed_code,
                'language': language
            }
        except Exception as e:
            return {'error': str(e)}

    def compare_files(self, submissions, query):
        processed_submissions = {}
        
        # queryies
        tokenizer = query.get('tokenizer', 'default')
        print("Comparing using tokenizer:", tokenizer)
        
        # Process all submissions
        for file_name, submission in submissions.items():
            result = self.process_submission(submission)
            if 'error' in result:
                return jsonify({
                    'error': f"Error processing {file_name}: {result['error']}", 
                    'traceback': result.get('traceback')
                }), 400
            print(f"file: {file_name} processed.")
            processed_submissions[file_name] = result
        
        filenames = list(processed_submissions.keys())
        file_count = len(filenames)
        
        weights = {
            'structural': 0.3,
            'token': 0.2,
            'tfidf': 0.2,
            'semantic': 0.3
        }
        
        comparison_results = []
        
        for i in range(file_count):
            code1 = processed_submissions[filenames[i]]['code']
            lang1 = processed_submissions[filenames[i]]['language']
            
            comparisons = []
            for j in range(file_count):
                if i != j:
                    code2 = processed_submissions[filenames[j]]['code']
                    lang2 = processed_submissions[filenames[j]]['language']
                    
                    if lang1 != lang2:
                        continue
                    
                    similarity_results = self.compute_similarity(
                        code1,
                        code2,
                        language=lang1,
                        weights=weights,
                        tokenizer=tokenizer
                    )
                    
                    comparisons.append({
                        "filename": filenames[j],
                        "structural_similarity": float(similarity_results['structural_similarity'] * 100),
                        "token_similarity": float(similarity_results['token_similarity'] * 100),
                        "tfidf_similarity": float(similarity_results['tfidf_similarity'] * 100),
                        "semantic_similarity": float(similarity_results['semantic_similarity'] * 100),
                        "combined_similarity": float(similarity_results['total_similarity'] * 100),
                        "potential_plagiarism": bool(similarity_results['total_similarity'] > 0.8)
                    })
            
            comparison_results.append({
                "file": filenames[i],
                "comparisons": comparisons
            })

        return comparison_results