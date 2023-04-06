# tetradat


## Description

Software product `tetradat` (**TE**nsor **TR**ain **AD**versarial **AT**tacks) for generation of adversarial examples for artificial neural networks (ANNs) using tensor train (TT) decomposition and optimizers based on it, i.e., [TTOpt](https://github.com/AndreiChertkov/ttopt) and [PROTES](https://github.com/anabatsh/PROTES) optimizers.


## Installation

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name tetradat python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate tetradat
    ```

4. Install dependencies:
    ```bash
    pip install jupyterlab "jax[cpu]==0.4.3" optax teneva==0.13.2 ttopt==0.5.0 protes==0.2.3 torch torchvision snntorch scikit-image matplotlib PyYAML nevergrad requests urllib3
    ```

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tetradat --all -y
    ```


## Usage

Run `python manager.py ARGS`, then see the outputs in the terminal and results in the `result` folder. Before starting the new calculation, you can completely delete or rename the `result` folder. A new `result` folder will be created automatically in this case.

> To run the code on the cluster, we used the `zhores_run.sh` bash script (in this case, the console output will be saved in a file `zhores_out.txt`).

Supported combinations of the `manager.py` script arguments:

- `python manager.py --data cifar10 --task check --kind data` TODO


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
