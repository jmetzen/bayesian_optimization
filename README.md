Summary
=======
This repository contains some basic code for Bayesian optimization

Installation
============

Install the current development version of scikit-learn (or sklearn version 0.18 once this is available)

    git clone git@github.com:scikit-learn/scikit-learn.git
    cd sklearn
    sudo python setup.py install

Install `bayesian_optimization`

    git clone git@git.hb.dfki.de:jmetzen/bayesian_optimization.git
    cd bayesian_optimization
    sudo python setup.py install


Usage
=====
Some usage examples are contained in the folder "examples". To reproduce the results from the ICML 2016 paper
"Minimum Regret Search for Single- and Multi-Task Optimization", please execute the jupyter notebook "examples/mrs_evaluation.ipynb." 

The directory bolero_bayes_opt contains modules which can be used with the external package BOLeRO, which is not yet open source and can thus be ignored for the moment.