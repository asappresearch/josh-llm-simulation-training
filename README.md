# Sparse Rewards Can Self-Train Dialogue Agents
Barrett Martin Lattimer, Varun Gangal, Ryan McDonald, Yi Yang

contact: blattimer@asapp.com

paper: tbd

This repo is currently being refactored (under the branch josh_refactor) which will be completed shortly. 

This repo runs both JOSH, the ToolWOZ, and τ-bench dataset. This repo also contains ways of logging training and preference-annotated episodes from user-simulator interactions and LORA-driven preference tuning of small LLMs from such preference annotated experience.


## Installation
1. Run the following in a new env
```
pip install -e .
```
2. Unzip the ```dataset.zip``` file in the ```data``` folder

## Quick Start
First, make sure you have your openai credentials in the environment
```
export OPENAI_API_KEY= # api_key
export OPENAI_ORGANIZATION= # api_org
```
If you're running Llama or another local model, you will also need to set HF_TOKEN much in the same way.

Then you can run a simulation by doing the following example
```
python josh/main.py
```

Wherever you see HF_KEY please replace it by your huggingface token.


## MT-Bench

(If you want to later evaluate MTBench)
```
unzip mtbencheval.zip
```
