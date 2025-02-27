from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer