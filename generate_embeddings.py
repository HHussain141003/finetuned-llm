import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, db_path: str = "microsoft_docs.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
    def load_documents(self, min_word_count: int = 50):
        """Load documents from database."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
            
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # First check what columns exist
            cursor = conn.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            logger.info(f"Available columns: {columns}")
            
            # Build query based on available columns
            base_query = "SELECT id, title, content"
            optional_cols = []
            
            if "url" in columns:
                optional_cols.append("url")
            if "category" in columns:
                optional_cols.append("category")
            if "word_count" in columns:
                optional_cols.append("word_count")
            
            if optional_cols:
                query = f"{base_query}, {', '.join(optional_cols)} FROM documents"
            else:
                query = f"{base_query} FROM documents"
            
            query += " WHERE content IS NOT NULL AND content != ''"
            
            if "word_count" in columns:
                query += " AND word_count >= ?"
                cursor = conn.execute(query, [min_word_count])
            else:
                query += " AND LENGTH(content) >= ?"
                cursor = conn.execute(query, [min_word_count * 5])
            
            rows = cursor.fetchall()
            
            self.documents = []
            for row in rows:
                row_dict = dict(row)
                
                doc = {
                    "id": row_dict["id"],
                    "title": row_dict.get("title") or "Untitled",
                    "content": row_dict["content"],
                    "category": row_dict.get("category", "unknown"),
                    "url": row_dict.get("url", ""),
                    "word_count": row_dict.get("word_count", len(row_dict["content"].split()))
                }
                self.documents.append(doc)
                
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        finally:
            conn.close()
    
    def chunk_document(self, content: str, max_words: int = 500) -> list:
        """Split long documents into chunks."""
        words = content.split()
        if len(words) <= max_words:
            return [content]
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        
        return chunks
    
    def process_documents(self):
        """Process documents and create chunks."""
        processed_docs = []
        
        for doc in self.documents:
            chunks = self.chunk_document(doc["content"])
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "id": f"{doc['id']}_chunk_{i}",
                    "original_id": doc["id"],
                    "title": doc["title"],
                    "content": chunk,
                    "category": doc["category"],
                    "url": doc["url"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                processed_docs.append(chunk_doc)
        
        self.documents = processed_docs
        logger.info(f"Created {len(self.documents)} document chunks")
    
    def generate_embeddings(self, batch_size: int = 32):
        """Generate embeddings for all documents."""
        if not self.model:
            self.load_model()
            
        if not self.documents:
            raise ValueError("No documents loaded")
        
        logger.info(f"Generating embeddings for {len(self.documents)} chunks...")
        
        # Extract text content
        texts = [doc["content"] for doc in self.documents]
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        
    def create_index(self):
        """Create FAISS index."""
        if self.embeddings is None:
            raise ValueError("No embeddings found")
        
        dimension = self.embeddings.shape[1]
        
        # Use simple flat index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def save_all(self, prefix: str = "microsoft_learn"):
        """Save index and documents."""
        if self.index is None:
            raise ValueError("No index created")
        
        # Save FAISS index
        index_file = f"{prefix}_index.faiss"
        faiss.write_index(self.index, index_file)
        
        logger.info(f"Saved files: {index_file}")
    
    def get_stats(self):
        """Get basic statistics."""
        if not self.documents:
            return {}
        
        categories = {}
        word_counts = []
        
        for doc in self.documents:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            word_counts.append(len(doc["content"].split()))
        
        return {
            "total_chunks": len(self.documents),
            "categories": categories,
            "avg_words_per_chunk": np.mean(word_counts) if word_counts else 0,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None
        }

def main():
    # Configuration
    DB_PATH = "microsoft_docs.db"
    MODEL_NAME = "all-MiniLM-L6-v2"
    MIN_WORD_COUNT = 50
    BATCH_SIZE = 32
    
    embedder = Embedder(DB_PATH, MODEL_NAME)
    
    try:
        embedder.load_documents(MIN_WORD_COUNT)
        embedder.process_documents()
        embedder.generate_embeddings(BATCH_SIZE)
        embedder.create_index()
        embedder.save_all()

        # Show stats
        stats = embedder.get_stats()
        logger.info("Processing complete!")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Categories: {list(stats['categories'].keys())}")
        logger.info(f"Avg words per chunk: {stats['avg_words_per_chunk']:.1f}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
