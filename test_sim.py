from time import perf_counter as tpc


from sim_wrapper import SimWrapper


if __name__ == '__main__':
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