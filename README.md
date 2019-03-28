# Summary

This repository contains some basic code for Bayesian optimization


## Installation

Install dependencies

1. Install [nlopt](https://github.com/stevengj/nlopt) for your Python version.
2. Install dependencies with pip: `[sudo] pip[3] install -r requirements.txt`
   (requirements.txt can be found in the repository)
3. (Optional:) install [BOLeRo](https://github.com/rock-learning/bolero)

Install `bayesian_optimization`

    git clone https://github.com/rock-learning/bayesian_optimization.git
    cd bayesian_optimization
    sudo python setup.py install


## Usage

Some usage examples are contained in the folder "examples". To reproduce the results from the ICML 2016 paper
"Minimum Regret Search for Single- and Multi-Task Optimization", please execute the jupyter notebook "examples/mrs_evaluation.ipynb." 

The directory bolero_bayes_opt contains modules which can be used with the external package BOLeRO, which is not yet open source and can thus be ignored for the moment.