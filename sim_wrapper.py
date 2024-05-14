from sentence_transformers import SentenceTransformer
from sentence_transformers import util


class SimWrapper:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def run(self, base, refs):
        is_many = isinstance(refs, list)
        if not is_many:
            refs = [refs]

        sentences = [base, refs]
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        scores = util.cos_sim(embeddings, embeddings)[0][1:]

        return scores if is_many else scores[0]