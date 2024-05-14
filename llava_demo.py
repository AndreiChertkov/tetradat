from time import perf_counter as tpc


from llava_wrapper import LlavaWrapper


if __name__ == '__main__':
    t = tpc()
    llava = LlavaWrapper()
    print(f'\n\nPREPARED | Time: {tpc()-t:-8.2f} sec')

    while True:
        image_file = input('Image:')
        prompt = input('Prompt:')

        t = tpc()
        result = llava.run(prompt, image_file)
        print(f'\n\nDONE    | Time: {tpc()-t:-8.2f} sec | Result :\n', result)