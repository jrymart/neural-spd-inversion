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

The python scripts and jupyter notebooks automatically generate from the org notebooks upon each commit.

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

## Components 

Each component is an org notebook, a jupyter notebook, and a python script.  Usually there is a corresponding SLURM script that runs the Python script but not always.  SLURM scripts will need to be updated with appropriate headers for your account and HPC.
| component   | description |
| ----------- | ----------- |
| 01_dataprep | downloads data from zenodo, calculates topographic derivatives |
| 02_training | trains the neural networks for the projects |
| 03_evaluation | evaluates the neural networks on the test set |
| 04a_valley_test | evaluates the neural networks on the valley spacing dataset |
| 04b_drainage_density_test | evaluates the neural networks on the drainage density dataset |
| 04c_shuffle_test | evaluates the neural network on the shuffled and swapped datasets |
| 04d_topographic_variation | calculates the between tile variance for the appendix |
| 05_analysis | calculates aggregate performance for the various tests |

The notebook `dataprep_training_eval_for_colab.ipynb` is combination of the first three components for use on Google Colab.
The script `all_plots.py` generates all plots.  It is a combination of all the plot notebooks in `notebooks/org/plots` (or their corresponding jupyter notebooks)
