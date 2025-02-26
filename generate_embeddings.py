import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("cleaned_json.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [entry["content"] for entry in data]
embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)