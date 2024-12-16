import os

def load_documents_from_folders(path, metadata):
    documents = []
    
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append({'text': text, 'metadata': metadata})
    
    return documents

