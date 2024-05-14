from time import perf_counter as tpc


from llava_wrapper import LlavaWrapper


if __name__ == '__main__':
    t = tpc()
    llava = LlavaWrapper()
    print(f'\n\nPREPARED | Time: {tpc()-t:-8.2f} sec')

    t = tpc()
    prompt = 'What are the things I should be cautious about when I visit here?'
    image_file = 'https://llava-vl.github.io/static/images/view.jpg'
    result = llava.run(prompt, image_file)
    print(f'\n\nDONE #1 | Time: {tpc()-t:-8.2f} sec | Result :\n', result)

    t = tpc()
    prompt = 'What do you see on this picture?'
    image_file = 'https://i.natgeofe.com/n/cad5d203-d715-4392-881c-3f33312652fe/00000169-ca0e-dfb8-a969-ea4e727d0002_3x2.jpg?wp=1&w=1436&h=958'
    result = llava.run(prompt, image_file)
    print(f'\n\nDONE #2 | Time: {tpc()-t:-8.2f} sec | Result :\n', result)