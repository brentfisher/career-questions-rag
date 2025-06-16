import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load the index
index = faiss.read_index('extracted_index.faiss')

id_map = []
feature_map = {}
with open('data/blogs.csv', 'r') as f:
    next(f)  # Skip header if exists
    for idx, line in enumerate(f):
        question = line.strip()
        id_map.append((int(idx), question))  # Or just question if you don't need the idx


with open('idmap.json', 'r') as f:
    feature_map = json.load(f)

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('tomaarsen/bge-base-en-v1.5')

# Example query
# query = "How do i earn $30 an hour?"
query = "what is the difference between a physicians assistant and a nurse?"

query_vector = model.encode([query], normalize_embeddings=True)

# Encode and normalize query_vector = model.encode([query])
faiss.normalize_L2(query_vector)

# Search top-k nearest neighbors
top_k = 5
distances, indices = index.search(query_vector, top_k)

for idx, distance in zip(indices[0], distances[0]):
    if (idx < 0 or idx >= len(feature_map)):
        print(f"Index {idx} out of bounds for feature_map.")
        continue
    else:
        feature_chunk = feature_map[str(idx)]
        print(f"{id_map[feature_chunk['blog_id']]}: {feature_chunk['chunk_text']} (Distance: {distance:.4f})")
        # matched_text = feature_map[str(idx)]
        # matched_question = id_map[idx][1]  # id_map[idx] is (ID, question)
        # print(f"Match: {matched_question} (Distance: {distance:.4f})")
