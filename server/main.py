# main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from detector_functions import EnhancedCodeSimilarityDetector  
import traceback


# import time for duration calculation
import time

app = Flask(__name__)
CORS(app)

# Initialize the detector once as a global variable
detector = EnhancedCodeSimilarityDetector()

class CustomJSONProvider(app.json_provider_class):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)

app.json = CustomJSONProvider(app)

def process_submission(submission):
    """
    Process a single submission using the enhanced detector
    """
    try:
        language = submission['language']
        code = submission['code']
        
        # Preprocess the code
        preprocessed_code = detector.preprocess_code(code, language)
        
        return {
            'code': preprocessed_code,
            'language': language
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/compare', methods=['POST'])
def compare_submissions():
    start_time = time.time()
    try:
        print("compare request received")
        if request.json is None:
            return jsonify({'error': 'No JSON payload received'}), 400
        
        data = request.json
        if 'submissions' not in data:
            return jsonify({'error': 'No submissions field in payload'}), 400
        
        submissions = data['submissions']
        processed_submissions = {}
        
        # Process all submissions
        for file_name, submission in submissions.items():
            result = process_submission(submission)
            if 'error' in result:
                return jsonify({
                    'error': f"Error processing {file_name}: {result['error']}", 
                    'traceback': result.get('traceback')
                }), 400
            print(f"file: {file_name} processed.")
            processed_submissions[file_name] = result
        
        # Get the list of filenames
        filenames = list(processed_submissions.keys())
        file_count = len(filenames)
        
        # Custom weights for similarity calculation
        weights = {
            'structural': 0.3,
            'token': 0.2,
            'tfidf': 0.2,
            'semantic': 0.3
        }
        
        comparison_results = []
        
        # Compare each file with every other file
        for i in range(file_count):
            code1 = processed_submissions[filenames[i]]['code']
            lang1 = processed_submissions[filenames[i]]['language']
            
            comparisons = []
            for j in range(file_count):
                if i != j:
                    code2 = processed_submissions[filenames[j]]['code']
                    lang2 = processed_submissions[filenames[j]]['language']
                    
                    # Skip comparison if languages don't match
                    if lang1 != lang2:
                        continue
                    
                    # Compute comprehensive similarity
                    similarity_results = detector.compute_similarity(
                        code1,
                        code2,
                        language=lang1,
                        weights=weights
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
        
        return jsonify(comparison_results)
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print(tb_str)
        return jsonify({
            'error': str(e), 
            'traceback': tb_str, 
            'request': request.json
        }), 500
    finally:
        print(f"Total time taken: {time.time() - start_time} seconds")
        
@app.route('/test', methods=['GET'])
def test():
    try:
        return jsonify({'message': 'Server is running with Enhanced Code Similarity Detector!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)