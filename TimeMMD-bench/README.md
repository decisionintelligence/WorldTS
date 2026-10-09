# VoT

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.10), install the dependencies with the following command:

```
conda create -n vot python==3.10
pip install -r requirements.txt
```
And you need to download the required language model in the `language_model` directory and set the corresponding `--llm_dim` parameter in `run_vot.py`

### Train and evaluate model

We provide the experiment scripts for all benchmarks under the folder `./scripts/vot.sh`. You can reproduce an experiment result as the following:

```shell
bash scripts/vot.sh
```
