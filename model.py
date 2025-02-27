from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("microsoft_learn_index.faiss")
with open("document_store.json", "r", encoding="utf-8") as f:
    document_store = json.load(f)

embedder = SentenceTransformer("./models/all-MiniLM-L6-v2")

model_name = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
