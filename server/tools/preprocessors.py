import re
import os

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

def process_files(directory):
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
                        tree_sequence = tree_to_sequence(preprocessed_code, language)
                        codebert_embedding = get_codebert_embedding(tree_sequence)
                        submission = {
                            'sequence': tree_sequence,
                            'language': language,
                            'embedding': codebert_embedding,
                            'tokens': set(tree_sequence.split())
                        }
                        submissions[file] = submission
                    except UnicodeDecodeError:
                        print(f"Error reading {file_path}. Skipping.")
            except ValueError as e:
                print(f"Skipping file {file}: {str(e)}")
    return submissions

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
    