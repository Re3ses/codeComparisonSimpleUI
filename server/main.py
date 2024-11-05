# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from functions import compute_similarities, preprocess_code, tree_to_sequence, get_codebert_embedding

app = Flask(__name__)
CORS(app)

class CustomJSONProvider(app.json_provider_class):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bool):  # Add explicit boolean handling
            return bool(obj)  # This ensures booleans are properly serialized
        return super().default(obj)

app.json = CustomJSONProvider(app)

def process_submission(submission):
    try:
        language = submission['language']
        code = submission['code']
        preprocessed_code = preprocess_code(code, language)
        tree_sequence = tree_to_sequence(preprocessed_code, language)
        codebert_embedding = get_codebert_embedding(tree_sequence)
        
        return {
            'sequence': tree_sequence,
            'language': language,
            'embedding': codebert_embedding,
            'tokens': set(tree_sequence.split())
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/compare', methods=['POST'])
def compare_submissions():
    try:
        if request.json is None:
            return jsonify({'error': 'No JSON payload received'}), 400
        
        submissions = request.json
        results = {}
        for file_name, submission in submissions.items():
            results[file_name] = process_submission(submission)
        
        codebert_similarities, jaccard_similarities, tfidf_similarities = compute_similarities(results)
        
        filenames = list(results.keys())
        file_count = len(filenames)
        
        codebert_results = []
        
        for i in range(file_count):
            comparisons = [
                {
                    "filename": filenames[j],
                    "codebert_similarity": float(codebert_similarities[i][j]),  # Convert to float
                    "jaccard_similarity": float(jaccard_similarities[i][j]),    # Convert to float
                    "tfidf_similarity": float(tfidf_similarities[i][j]),        # Convert to float
                    "combined_similarity": float((codebert_similarities[i][j] + jaccard_similarities[i][j] + tfidf_similarities[i][j]) / 3),
                    "potential_plagiarism": bool((codebert_similarities[i][j] + jaccard_similarities[i][j] + tfidf_similarities[i][j]) / 3 > 80)
                }
                for j in range(file_count) if i != j
            ]
            codebert_results.append({"file": filenames[i], "comparisons": comparisons})
            
        return jsonify(codebert_results)  # Add jsonify here
    
    except Exception as e:
        submissions = request.json
        out = []
        for file_name, submission in submissions.items():
            out.append({file_name: submission})
            
        return jsonify({'error': str(e), 'request': request.json }), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        return jsonify({'message': 'Server is running!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)