# Install required packages (if not installed)
# pip install transformers torch scikit-learn matplotlib

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Qwen3-Embedding model from Hugging Face
model_name = 'Qwen/Qwen3-Embedding-0.6B'  # Replace with actual model repo if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example sentences with some similarities
# sentences = [
#     "The quick brown fox jumps over the lazy dog.",
#     "A fast dark-colored fox leaps above a sleeping canine.",
#     "I love to watch movies on weekends with friends.",
#     "Watching films with my buddies is my favorite weekend activity.",
#     "The rain in Spain stays mainly in the plain.",
#     "Machine learning models are useful for text similarity.",
#     "Artificial intelligence techniques can measure sentence similarity.",
#     "Cats are often found sleeping in the sun.",
#     "the slow brown cat rolls in the dirt"
# ]

df = pd.read_csv('sentences.csv')
sentences = df['sentence'].tolist()


# Tokenize and compute embeddings
def get_embeddings(sent_list):
    inputs = tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        # Use the EOS token for sentence embeddings
        eos_mask = inputs['attention_mask'].ne(0).long().sum(dim=1) - 1
        batch_indices = torch.arange(out.last_hidden_state.size(0))
        embeddings = out.last_hidden_state[batch_indices, eos_mask]
    return embeddings.numpy()


embeddings = get_embeddings(sentences)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Reduce embeddings to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the sentence embeddings
plt.figure(figsize=(12, 10))
x = reduced_embeddings[:, 0]
y = reduced_embeddings[:, 1]
plt.scatter(x, y, color='blue')

# Annotate points with sentence fragments
# for i, sentence in enumerate(sentences):
#     plt.annotate(sentence[:25] + '...', (x[i], y[i]), fontsize=9)

#shorter names
for i, sentence in enumerate(sentences):
    label = f"{i}"
    # label = f"{[i]}: {sentence[:25]}..."
    plt.annotate(sentence[:40] + '...', (x[i], y[i]), fontsize=5)

plt.title('Sentence Embeddings Visualization (Qwen3-Embedding)')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.grid(True)
plt.show()

# Optionally, plot the similarity matrix
# import seaborn as sns
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=False, yticklabels=False)
# plt.title('Cosine Similarity Matrix')
# plt.show()



