from sentence_transformers import SentenceTransformer
from sentence_transformers import util


class SimWrapper:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def run(self, base, refs):
        is_many = isinstance(refs, list)
        if not is_many:
            refs = [refs]

        embeds = self.model.encode([base] + refs, convert_to_tensor=True)
        scores = util.cos_sim(embeds, embeds)[0][1:].detach().cpu().numpy()

        return scores if is_many else scores[0]