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
device = "cpu"

def retrieve_documents(query, top_k=5):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    print(f"Retrieved documents: {[document_store[i] for i in indices[0]]}")
    return [document_store[i] for i in indices[0]]

def generate_answer(query):
    retrieved_docs = retrieve_documents(query)
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    prompt = f"""
    Answer the question in simple and concise terms using the context and Microsoft Learn material only.
    
    Context:
    {context}

    Question: {query}
    
    Please provide a detailed and step-by-step answer, including any relevant code or examples.
    """

    input_data = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = input_data.input_ids.to(device)
    attention_mask = input_data.attention_mask.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output = model.generate(input_ids, max_new_tokens=500, do_sample=True, temperature=0.3)
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = input("Question: ")
answer = generate_answer(question)
print(f"Answer: {answer}")