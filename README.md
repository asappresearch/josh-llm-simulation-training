# josh-llm-simulation-training

This repo is meant to run simulations of the transformed multiwoz dataset. You can test your own agents by making 
an AgentSimulator object.

This repo also contains ways of logging training and preference-annotated episodes from user-simulator interactions and LORA-driven preference tuning of small LLMs from such preference annotated experience.

## Pre-installation

(If you want to later evaluate MTBench)
```
unzip mtbencheval.zip
```

## Installation
Run the following in a new env
```
pip install -e .
```
Download the ```data.json``` and ```delex.json``` files from the following link and place them in the ```data``` folder: https://drive.google.com/drive/folders/1FZmirZ6m9i769KyPEzS69DZoO25lqK-u?usp=sharing

## Quick Start
First, make sure you have your openai credentials in the environment
```
export OPENAI_API_KEY= # api_key
export OPENAI_ORGANIZATION= # api_org
```
You also need to set HF_TOKEN much in the same way.

Then you can run a simulation by doing the following
```
python multiwoz_api/main.py
```

Wherever you see HF_KEY please replace it by your huggingface token.
