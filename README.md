# tetradat


## Description

Software product `tetradat` (**TE**nsor **TR**ain **AD**versarial **AT**tacks) for generation of adversarial examples for artificial neural networks (ANNs) using tensor train (TT) decomposition and optimizer based on it, i.e., [PROTES](https://github.com/anabatsh/PROTES) optimizer.


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
    pip install jupyterlab "jax[cpu]==0.4.6" optax teneva==0.14.1 ttopt==0.5.0 protes==0.3.1 torch torchvision snntorch scikit-image matplotlib nevergrad requests urllib3
    ```
    > Run also `pip install git+https://github.com/jeromerony/adversarial-library`

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name tetradat --all -y
    ```


## Usage

Run `python manager.py ARGS`, then see the outputs in the terminal and results in the `result` folder. Before starting the new calculation, you can completely delete or rename the `result` folder. A new `result` folder will be created automatically in this case.

> To run the code on the cluster, we used the `zhores_run.sh` bash script (in this case, the console output will be saved in a file `zhores_out.txt`).


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
