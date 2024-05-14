from time import perf_counter as tpc


from llava_wrapper import LlavaWrapper


def demo():
    t = tpc()
    llava = LlavaWrapper()
    print(f'\n\nPREPARED | Time: {tpc()-t:-8.2f} sec')

    img = ''
    txt = ''

    for i in range(100000):
        print('\n\n' + '-'*50 + '\n' + f'--- DEMO # {i+1:-4d}')
        
        img = input('Image  > ') or img
        txt = input('Prompt > ') or txt

        if txt == 'END':
            break

        if not img:
            print('OOPS! Please provide the path to image')
        if not txt:
            print('OOPS! Please provide the prompt')

        t = tpc()
        result = llava.run(img, txt)
        print(f'\n\nDONE    | Time: {tpc()-t:-8.2f} sec | Result :\n', result)


def test():
    t = tpc()
    llava = LlavaWrapper()
    print(f'\n\nPREPARED | Time: {tpc()-t:-8.2f} sec')

    t = tpc()
    img = 'https://llava-vl.github.io/static/images/view.jpg'
    txt = 'What are the things I should be cautious about when I visit here?'
    result = llava.run(img, txt)
    print(f'\n\nDONE #1 | Time: {tpc()-t:-8.2f} sec | Result :\n', result)

    t = tpc()
    img = 'https://i.natgeofe.com/n/cad5d203-d715-4392-881c-3f33312652fe/00000169-ca0e-dfb8-a969-ea4e727d0002_3x2.jpg?wp=1&w=1436&h=958'
    txt = 'What do you see on this picture?'
    result = llava.run(img, txt)
    print(f'\n\nDONE #2 | Time: {tpc()-t:-8.2f} sec | Result :\n', result)


if __name__ == '__main__':
    print('\n\n --- TEST --- \n\n')
    test()
    print('\n\n --- DEMO --- \n\n')
    demo()
