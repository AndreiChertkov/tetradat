from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from time import perf_counter as tpc


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


def _demo():
    t = tpc()
    sim = SimWrapper()
    print(f'\n\nPREPARED | Time: {tpc()-t:-8.2f} sec')

    t = tpc()
    base = 'The cat sits outside'
    ref = 'The cat sits here'
    score = sim.run(base, ref)
    print(f'\n\nDONE #1 | Time: {tpc()-t:-8.2f} sec | Result :\n', score)

    t = tpc()
    base = 'The cat sits outside'
    refs = ['The cat sits here', 'The cat sits at home', 'The cat sits outside']
    scores = sim.run(base, refs)
    print(f'\n\nDONE #2 | Time: {tpc()-t:-8.2f} sec | Result :\n', scores)


if __name__ == '__main__':
    _demo()