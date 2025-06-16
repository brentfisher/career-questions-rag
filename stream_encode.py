from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import csv
import json
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Setup
# model = SentenceTransformer('all-MiniLM-L6-v2')
# index = faiss.IndexFlatIP(384)  # Inner Product (Cosine similarity if you normalize)
model = SentenceTransformer('tomaarsen/bge-base-en-v1.5')
index = faiss.IndexFlatIP(768)  # Inner Product (Cosine similarity if you normalize)
faiss.normalize_L2  # We will normalize before adding

batch_size = 512
blog_texts = []
global_chunk_id = 0  # Must match FAISS index order
id_map = {}  # Optional: store IDs or mapping indices

def chunk_text(text, chunk_size=300, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous chunk
            current_chunk = " ".join(current_chunk.strip().split()[-overlap // 5:])  # Approx. 5 chars per word
            current_chunk += " " + sentence

    if current_chunk:
        chunk = current_chunk.strip()
        chunks.append(chunk)

    #save the chunks with their blog id
    return chunks

def index_embeddings(chunk, blog_id, chunk_id):
    embedding = model.encode(chunk, normalize_embeddings=True)
    embedding = embedding.reshape(1, -1)
    index.add(embedding)
    # faiss.normalize_L2(embeddings)

    # Save mapping from FAISS index to blog and chunk
    id_map[chunk_id] = {'blog_id': blog_id, 'chunk_text': chunk}


# with open('blog_summary.csv', newline='') as csvfile:
with open('blog_output.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for blog_id, row in enumerate(reader):
        blog = row[0]
        chunks = chunk_text(blog)

        for chunk in chunks:
            print(f"Indexing chunk {global_chunk_id} from blog {blog_id}")
            index_embeddings(chunk, blog_id, global_chunk_id)
            global_chunk_id += 1


# Save index to disk
faiss.write_index(index, 'extracted_index.faiss')

#write id_map to json
with open('idmap.json', 'w') as f:
    json.dump(id_map, f)

