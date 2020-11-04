# FLSim

## About

Welcome to **FLSim**, a PyTorch based federated learning simulation framework, created for experimental research in a paper accepted by [IEEE INFOCOM 2020](https://infocom2020.ieee-infocom.org):

[Hao Wang](https://www.haow.ca), Zakhary Kaplan, [Di Niu](https://sites.ualberta.ca/~dniu/Homepage/Home.html), [Baochun Li](http://iqua.ece.toronto.edu/bli/index.html). "Optimizing Federated Learning on Non-IID Data with Reinforcement Learning," in the Proceedings of IEEE INFOCOM, Beijing, China, April 27-30, 2020.



## Installation

To install **FLSim**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLSim** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `fl-py37` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `fl-py37` environment with:

```shell
conda activate fl-py37
```

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

##### `run.py` flags

* `--config` (`-c`): path to the configuration file to be used.
* `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.

##### `config.json` files

**FLSim** uses a JSON file to manage the configuration parameters for a federated learning simulation. Provided in the repository is a generic template and three preconfigured simulation files for the CIFAR-10, FashionMNIST, and MNIST datasets.

For a detailed list of configuration options, see the [wiki page](https://github.com/iQua/flsim/wiki/Configuration).

If you have any questions, please feel free to contact Hao Wang (haowang@ece.utoronto.ca)
