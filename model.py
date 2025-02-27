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

def retrieve_documents(query, top_k=5):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [document_store[i] for i in indices[0]]

def generate_answer(query):
    retrieved_docs = retrieve_documents(query)
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    prompt = f"Answer the question based only on the following Microsoft Learn content:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=20000)
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = "How do I configure Azure Active Directory authentication?"
answer = generate_answer(question)
print(f"Answer: {answer}")