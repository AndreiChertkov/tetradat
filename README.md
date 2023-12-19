# tetradat


## Description

Package `tetradat` (**TE**nsor **TR**ain **AD**versarial **AT**tacks) for generation of adversarial examples for artificial neural networks from computer vision domain using tensor train (TT) decomposition and PROTES optimizer based on it. Please, see [teneva](https://github.com/AndreiChertkov/teneva) and [PROTES](https://github.com/anabatsh/PROTES) repositories for details.


## Installation

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org) (version 3.8);

2. Create a virtual environment:
    ```bash
    conda create --name tetradat python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate tetradat
    ```

4. Install dependencies:
    - To run the code on CPU device:
        ```bash
        pip install numpy matplotlib requests urllib3 protes==0.3.6 torch==1.12.1 torchvision==0.13.1 torchattacks==3.4.0
        ```
    - To run the code on GPU device, please see `zhores.py` script.

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tetradat --all -y
    ```


## Usage

Run `python manager.py ARGS`, then see the outputs in the terminal and results in the `result` folder. Before starting the new calculation, you can completely delete or rename the `result` folder; a new `result` folder will be created automatically in this case.

The calls with the following `ARGS` may be performed:

- To check the data:
    - `python manager.py --task check --kind data --data imagenet`

- To check the models:
    - `python manager.py --task check --kind model --data imagenet --model alexnet`
    - `python manager.py --task check --kind model --data imagenet --model googlenet`
    - `python manager.py --task check --kind model --data imagenet --model inception`
    - `python manager.py --task check --kind model --data imagenet --model mobilenet`
    - `python manager.py --task check --kind model --data imagenet --model resnet`
    - `python manager.py --task check --kind model --data imagenet --model vgg`

- To run attacks with the proposed TETRADAT method:
    - `python manager.py --task attack --kind attr --data imagenet --model googlenet --model_attr vgg`
        > You may use any of the models from above here. We select the helper model for attribution by `model_attr` argument.

- To run attacks with the baselines:
    - `python manager.py --task attack --kind bs_onepixel --data imagenet --model googlenet --model_attr vgg`
        > You may use any of the models from above here; and `bs_onepixel` or `bs_pixle`. We set `model_attr` argument above to skip the images for which the `vgg` model fails.

- To run the demo attack with the proposed TETRADAT method:
    - `python manager.py --task check --kind demo --data imagenet`

- To present the computation results:
    - `python show.py`
        > Data for tables will be in the console output and images will be in `result/_show` folder.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
