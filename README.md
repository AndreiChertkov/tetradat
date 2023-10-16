# tetradat


## Description

Software product `tetradat` (**TE**nsor **TR**ain **AD**versarial **AT**tacks) for generation of adversarial examples for artificial neural networks (ANNs) from computer vision domain using tensor train (TT) decomposition and optimizer based on it, i.e., [PROTES](https://github.com/anabatsh/PROTES) optimizer.


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
        pip install teneva_opti==0.5.1 torch==1.12.1 torchvision==0.13.1 matplotlib requests urllib3 torchattacks==3.4.0
        ```
    - To run the code on GPU device (we used zhores cluster, see `zhores_run.sh` script):
        ```bash
        pip install teneva_opti==0.5.0 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113 && pip install triton
        ```
    > In the case of problems with `scikit-learn`, uninstall it as `pip uninstall scikit-learn -y` and then install it from the anaconda: `conda install -c anaconda scikit-learn`.

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tetradat --all -y
    ```


## Usage

Run `python manager.py ARGS`, then see the outputs in the terminal and results in the `result` folder. Before starting the new calculation, you can completely delete or rename the `result` folder; a new `result` folder will be created automatically in this case.

The calls with the following `ARGS` may be performed:

- `python manager.py --task check --kind data --data imagenet`

- `python manager.py --task check --kind model --data imagenet --model alexnet`

- `python manager.py --task check --kind model --data imagenet --model vgg16`

- `python manager.py --task check --kind model --data imagenet --model vgg19`

- `python manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg16`

- `python manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg19`

- `python manager.py --task attack --kind attr --data imagenet --model vgg16 --model_attr alexnet`

- `python manager.py --task attack --kind attr --data imagenet --model vgg16 --model_attr vgg19`

- `python manager.py --task attack --kind attr --data imagenet --model vgg19 --model_attr alexnet`

- `python manager.py --task attack --kind attr --data imagenet --model vgg19 --model_attr vgg16`

- `python manager.py --task attack --kind bs1 --data imagenet --model alexnet`

> To run the code on the GPU-cluster, we used the `zhores_run.sh` bash script (in this case, the console output will be saved in a file `zhores_out.txt`).


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
