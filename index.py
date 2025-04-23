import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class IndexSearcher:
    def __init__(self, index_path="microsoft_learn_index.faiss", doc_store_path="document_store.json"):
        print(f"Loading document store from {doc_store_path}...")
        with open(doc_store_path, 'r', encoding='utf-8') as f:
            self.document_store = json.load(f)
        print(f"Loaded {len(self.document_store)} documents")
            
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Index contains {self.index.ntotal} vectors of dimension {self.index.d}")
        
        self.embedder = SentenceTransformer('models/all-MiniLM-L6-v2')
        # self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        print("Model loaded successfully")
    
    def search(self, query, top_k=5):
        # Encode the query
        query_embedding = self.embedder.encode([query])
        
        # Search the index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Get the documents
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.document_store) and idx >= 0:  # Ensure valid index
                doc = self.document_store[int(idx)]
                # Add distance score to the document
                doc_with_score = dict(doc)
                doc_with_score['search_score'] = float(1.0 - distance)  # Convert distance to similarity score
                doc_with_score['rank'] = i + 1
                results.append(doc_with_score)
            else:
                print(f"Warning: Invalid index {idx}")
        
        return results
    
if __name__ == "__main__":
    searcher = IndexSearcher()  
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
                  
        top_k = 5     
        results = searcher.search(query, top_k=top_k)       
        print(results)