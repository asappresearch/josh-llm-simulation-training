# Sparse Rewards Can Self-Train Dialogue Agents
Barrett Martin Lattimer, Varun Gangal, Ryan McDonald, Yi Yang

contact: blattimer@asapp.com

paper: https://arxiv.org/abs/2409.04617

This repo runs JOSH, the ToolWOZ, and τ-bench dataset. This repo also contains ways of logging training and preference-annotated episodes from user-simulator interactions and LORA-driven preference tuning of small LLMs from such preference annotated experience.


## Setup
1. Run the following in a new env
```
pip install -e .
```
2. Unzip the ```dataset.zip``` file in the ```data``` folder
   
3. Set up your openai credentials
```
export OPENAI_API_KEY= # api_key
export OPENAI_ORGANIZATION= # api_org
```
If you're running Llama or another local model, you will also need to set HF_TOKEN much in the same way. Wherever you see HF_KEY please replace it by your huggingface token.

## Running ToolWOZ

You can run ToolWOZ normally by doing the following
```
python josh_train/main.py
```
Increase the ```--max_concurrency``` depending on your api rate limits
### JOSH on ToolWOZ
Enable JOSH on ToolWOZ by adding the ```--josh``` flag, and make the running of JOSH print updates by also adding ```--josh_debug```

One example of a more involved JOSH prompt would be the following
```
python josh_train/main.py --josh --josh_debug --max_concurrency 20 --seed 20 --task_split train --temperature 1.0 --agent_strategy react --user_mode goal --model gpt-4o-mini --end_index 10 --beam_size 8
```

## Running τ-bench

We have added a clone of [τ-bench](https://github.com/sierra-research/tau-bench) to this repo with two run files, one for normal τ-bench testing and another for JOSH rollouts on τ-bench

To run τ-bench normally you can do
```
python tau-bench-eval/run.py
```

### JOSH on τ-bench
To run JOSH on τ-bench you can do
```
python tau-bench-eval/run.py --josh --debug
```

## Using JOSH
A class of JOSH is provided in this repo to be very flexible and work for a wide variety of user/agent interactions. To use JOSH yourself, you can start with the following code snippet
```
from josh_train.josh import JOSH, BaseJOSHAgent, BaseRewards, BaseJOSHUser
def add_error_message(agent):
        agent.messages.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return agent
    
def step_agent(agent:BaseJOSHAgent, **kwargs):
    pass_to_customer = agent.step(**kwargs)
    return agent, pass_to_customer

def step_user(user:BaseJOSHUser, agent:BaseJOSHAgent):
    agent, end_conversation = user.step(agent)
    return agent, end_conversation

josh = JOSH(
            rewards=BaseRewards(['say hello', 'say hello', 'say hello']),
            agent_step=step_agent,
            user_step=step_user,
            add_error_message=add_error_message,
            root_agent = BaseJOSHAgent(),
            user = BaseJOSHUser(),
            debug=True
        )

for _ in range(10):
    max_reward, all_done = josh.step()
    if all_done:
        break

print(max_reward)
print(josh.training_examples)
```

All classes can be built on top of, and expanded for further use.


## MT-Bench

(If you want to later evaluate MTBench)
```
unzip mtbencheval.zip
```

## Citation
Please cite if you enjoyed this work!
```
@article{lattimer2024sparse,
  title={Sparse Rewards Can Self-Train Dialogue Agents},
  author={Lattimer, Barrett Martin and Gangal, Varun and McDonald, Ryan and Yang, Yi},
  journal={arXiv preprint arXiv:2409.04617},
  year={2024}
}
```
