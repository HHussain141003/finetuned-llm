import os
import sqlite3
import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load .env and login
load_dotenv()
hf_token = os.getenv("HF_Login")
if not hf_token:
    raise ValueError("HF_Login not found in .env file")
login(hf_token)

print("Loading FAISS index and database...")
# Load FAISS index and connect to database
index = faiss.read_index("microsoft_learn_index.faiss")
db_path = "microsoft_docs.db"

# Test database connection
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.execute("SELECT COUNT(*) FROM documents")
doc_count = cursor.fetchone()[0]
conn.close()

print(f"Connected to database with {doc_count} documents")

print("Loading embedding model...")
# Load embedding model (same as used for generating embeddings)
embedder = SentenceTransformer("./models/all-MiniLM-L6-v2")

print("Loading tokenizer and model...")
# Configure quantization to improve performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8
)

# Load model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

print("Model loaded successfully!")

def get_document_by_id(doc_id):
    """Get document from database by ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.execute("SELECT * FROM documents WHERE id = ?", [doc_id])
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
        
    finally:
        conn.close()

def retrieve_documents(query, top_k=5):
    """Retrieve relevant documents using FAISS similarity search."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    retrieved_docs = []
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                # Get document by row number (FAISS index corresponds to document order)
                cursor = conn.execute("SELECT * FROM documents LIMIT 1 OFFSET ?", [int(idx)])
                row = cursor.fetchone()
                
                if row:
                    doc = dict(row)
                    doc['similarity_score'] = float(distance)
                    retrieved_docs.append(doc)
    
    finally:
        conn.close()
    
    return retrieved_docs

def generate_answer(query):
    """Generate answer using retrieved documents from database."""
    print(f"Processing query: {query}")
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, top_k=5)
    
    if not retrieved_docs:
        return "No relevant documents found in the Microsoft Learn knowledge base."
    
    print(f"Retrieved {len(retrieved_docs)} document chunks")
    
    # Build context from database documents
    context_parts = []
    for doc in retrieved_docs:
        title = doc.get('title', 'Untitled Document')
        content = doc.get('content', '')
        category = doc.get('category', '')
        
        # Show category for better context
        header = f"[{category}] {title}" if category else title
        
        # Limit content length
        if len(content) > 800:
            content = content[:800] + "..."
        
        context_parts.append(f"{header}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""[INST] You are a Microsoft documentation expert. Answer the question using ONLY the provided Microsoft Learn documentation. If the answer isn't in the documentation, say so clearly.

Microsoft Learn Documentation:
{context}

Question: {query}

Answer based only on the documentation above: [/INST]"""

    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        input_length = input_ids.shape[1]
        
        print("Generating response...")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer if answer else "I couldn't generate a proper response."
        
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {str(e)}"

def show_retrieved_docs(query):
    """Show what documents are being retrieved from the database."""
    docs = retrieve_documents(query, top_k=3)
    print(f"\nRetrieved from database for: '{query}'")
    
    for i, doc in enumerate(docs, 1):
        title = doc.get('title', 'No title')
        category = doc.get('category', 'Unknown')
        url = doc.get('url', '')
        
        print(f"\n{i}. [{category}] {title}")
        print(f"   URL: {url}")
        print(f"   Similarity: {doc.get('similarity_score', 'N/A'):.4f}")
        
        content_preview = doc.get('content', '')[:150] + "..." if len(doc.get('content', '')) > 150 else doc.get('content', '')
        print(f"   Preview: {content_preview}")

if __name__ == "__main__":
    while True:
        query = input("\nQuestion (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Show retrieved documents
        show_retrieved_docs(query)
        
        print("\n" + "="*50)
        
        # Generate answer
        answer = generate_answer(query)
        print(f"\nAnswer: {answer}")
        print("="*50)
