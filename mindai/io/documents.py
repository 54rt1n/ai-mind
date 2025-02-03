# mindai/io/documents.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
from typing import List, Dict, Tuple
import chardet

class Library:
    """
    Wraps a directory of documents, and provides a way to query them.
    """
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir
        self.extensions = {'org', 'txt', 'md', 'py', 'js', 'ts', 'json', 'yaml', 'yml', 'csv', 'log', 'xml', 'html', 'css', 'svelte'}

    @property
    def list_documents(self) -> List[Tuple[str, float, int]]:
        """
        A list of documents in the library.

        Returns:
        A list of tuples, each containing the document name, last modified time, and file size.
        """
        files = []
        for f in os.listdir(self.documents_dir):
            for ext in self.extensions:
                last_modified = os.path.getmtime(os.path.join(self.documents_dir, f))
                file_size = os.path.getsize(os.path.join(self.documents_dir, f))
                if f.endswith(ext):
                    files.append((f, last_modified, file_size))
                    break
        return files

    def exists(self, document_name: str) -> bool:
        """
        Check if a document exists in the library.
        """
        return os.path.exists(os.path.join(self.documents_dir, document_name))

    def read_document(self, document_name: str) -> str:
        """
        Read a document from the library, fixing any encoding issues.
        """
        file_path = os.path.join(self.documents_dir, document_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document '{document_name}' not found in the library.")
        
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
        
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            return raw_data.decode('utf-8', errors='replace')

    def search_documents(self, query: str) -> Dict[str, List[str]]:
        """
        Search for a query across all documents in the library.
        Returns a dictionary with document names as keys and lists of matching lines as values.
        """
        results = {}
        for doc_name, _, _ in self.list_documents:
            content = self.read_document(doc_name)
            matching_lines = [line.strip() for line in content.split('\n') if query.lower() in line.lower()]
            if matching_lines:
                results[doc_name] = matching_lines
        return results
