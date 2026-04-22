# neural-spd-inversion
Code to replicate the analysis for "Can Neural Networks Think Like Geomorphologists?"

## Overview
This repository contains code to
1. Download and process Streampower-Diffusion (SPD) landscape evolution model results
2. Train neural networks to infer model parameters from output topography (and derivatives)
3. Analyze the neural network's performance and learning
4. Generate figures for the paper "Can Neural Networks Think Like Geomorphologists?"

The code can be run in a variety of ways:
- python scripts
- SLURM scripts (which call the python scripts)
- jupyter notebooks
- a single jupyter notebook for use with colab 
- org notebooks for use with org-babel/emacs

## Dependency/environment Setup
A `pyproject.toml` file exists which outlines dependencies.  There are external dependency, one bundled dependcy (in the `lib` directory) and the python project itself (`neural_spd`) which contains a configuration file and helper functions.  This can be setup trivially with uv.  
uv can be installed with the following two commands in a bash compatable shell

``` sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```
then, from the project directory, simply run
``` sh
uv sync
```
to download dependencies and install the project in a virtual environment (which will be located in a created `.venv` directory)
There is an optional dependency `pygraphviz` which is only used to create one figure in the appendix.  To install, run
``` sh
uv sync --extra appendix
```
`pygraphviz` requires the graphviz C library to run, which cannot be installed and setup with uv.

## Configuration
The project can be configuration through the file `src/neural_spd/config.py`.  There are a variety of options here that can be changed, primarily to change what topographic derivatives the networks are trained on, and things like the epochs, learning rate, and batch size of the training.  Changing these, and other variables in this file fundamentally change the results of this study, and so should be changed with full understanding of the workflow and code.  If you are interested in adapting this for a similar but distinct experiment, please feel free to contact jo.martin@colorado.edu for assistance.



