import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class IndexSearcher:
    def __init__(self, index_path="microsoft_learn_index.faiss", doc_store_path="document_store.json"):
        # Load the document store
        print(f"Loading document store from {doc_store_path}...")
        with open(doc_store_path, 'r', encoding='utf-8') as f:
            self.document_store = json.load(f)
        print(f"Loaded {len(self.document_store)} documents")
            
        # Load the FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Index contains {self.index.ntotal} vectors of dimension {self.index.d}")
        
        # Load the embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('models/all-MiniLM-L6-v2')
        # self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        print("Model loaded successfully")
    
    def search(self, query, top_k=5):
        """
        Search for the most relevant documents for the given query.
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return (default: 5)
            
        Returns:
            list: List of dictionaries containing the most relevant documents
        """
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
    
    def display_results(self, results):
        """
        Display search results in a readable format
        """
        if not results:
            print("No matching results found.")
            return
            
        print(f"Found {len(results)} relevant results:\n")
        
        for result in results:
            print(f"Result #{result['rank']}):")
            
            # Get the document
            doc = result['document']
            
            # Display document details
            print(f"Title: {doc.get('title', 'No title')}")
            
            # Display a snippet of the content
            content = doc.get('content', 'No content')
            snippet = content[:200] + "..." if len(content) > 200 else content
            print(f"Content snippet: {snippet}")
            
            # Display URL if available
            if 'url' in doc:
                print(f"URL: {doc['url']}")
                
            print("-" * 80)
    
if __name__ == "__main__":
    # Create the searcher
    searcher = IndexSearcher()
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        # Set how many results to return
        top_k = 5
        
        # Perform the search
        results = searcher.search(query, top_k=top_k)
        
        # Display the results
        # searcher.display_results(results)
        print(results)