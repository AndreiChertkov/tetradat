from sentence_transformers import SentenceTransformer
from sentence_transformers import util


model = SentenceTransformer('all-MiniLM-L6-v2')


sentences = [
    'The cat sits outside',
    'A man is playing guitar',
    'I love pasta',
    'The new movie is awesome',
    'The cat plays in the garden',
    'A woman watches TV',
    'The new movie is so great',
    'Do you like pizza?',
]

# Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

# Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(cosine_scores.shape[0]):
    for j in range(cosine_scores.shape[1]):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

# Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print('{} \t\t {} \t\t Score: {:.4f}'.format(
        sentences[i], sentences[j], pair['score']
    ))