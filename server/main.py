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

# class CustomJSONProvider(app.json_provider_class):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, set):
#             return list(obj)
#         if isinstance(obj, bool):
#             return bool(obj)
#         return super().default(obj)

# app.json = CustomJSONProvider(app)

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

        comparison_results = detector.compare_files(submissions)
        
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