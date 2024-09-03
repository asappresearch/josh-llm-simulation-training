from multiwoz_api.utils import *
import openai
from openai import OpenAI
from user_simulator import UserSimulator
from agent_simulator_pref_tree import AgentSimulator
from conversation_state_pref_tree import Conversation
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from huggingface_hub import login
cwd = os.getcwd()

torch.cuda.empty_cache()

creds = get_openai_creds()
openai.api_key = creds['openai_key']
openai.organization = creds['openai_org']
api_key = creds['openai_key']
api_org = creds['openai_org']
client = OpenAI(api_key=api_key, organization=api_org)

hf_creds = get_hf_creds()
login(token=hf_creds["hf_token"])


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.eval()


dbs = create_dbs()
if not os.path.isfile('data/ground_truth_apis.json'):
    apis = create_apis(dbs)
    with open('data/ground_truth_apis.json', 'w') as file:
        json.dump(apis, file, indent=2)
else:
    with open('data/ground_truth_apis.json', 'r') as file:
        apis = json.load(file)

if not os.path.isfile('data/api_examples.json'):
    api_examples = create_api_examples(apis)
    with open('data/api_examples.json', 'w') as file:
        json.dump(api_examples, file, indent=2)
else:
    with open('data/api_examples.json', 'r') as file:
        api_examples = json.load(file)

for k in dbs.keys():
    dbs[k].close()

with open('data/valid_api_defs.json', 'r') as file:
    valid_api_defs = json.load(file)

with open('data/delex.json') as outfile:
    delex = json.load(outfile)

with open('data/data.json') as outfile:
    real_convos = json.load(outfile)


with open('data/testListFile.json', 'r') as file:
    test_ids = file.readlines()
test_ids = [x.strip() for x in test_ids]
test_ids = list(set(test_ids).intersection(apis.keys()))
with open('data/valListFile.json', 'r') as file:
    val_ids = file.readlines()
val_ids = [x.strip() for x in val_ids]
val_ids = list(set(val_ids).intersection(apis.keys()))

train_ids = list(set([k for k in apis.keys()]) - set(test_ids) - set(val_ids))

set_to_run = sorted(train_ids)


results = []
failed_apis=[]
training_paths_total = []
for idx, convo_key in enumerate(set_to_run):
    conversation_state = Conversation(convo_key, apis, delex)
    user = UserSimulator(goals=conversation_state.goals, modelname='gpt-4o')
    agent = AgentSimulator(api_examples, valid_api_defs, model=model, tokenizer=tokenizer, conversation_state=conversation_state)

    # Run the simulation
    conversations = []
    conversations_total = []
    training_paths_total = []
    percent_100_reached_final=False
    count = 0
    while count<10:
        print(count)
        replies = []
        if len(conversations)==0:
            replies = [user.step([], client)]
        else:
            for messages in conversations:
                replies.append(user.step(messages, client))

        training_paths, conversations, percent_100_reached = agent.step(replies)
        if not training_paths and not conversations:
            break
        conversations_total = conversations
        percent_100_reached_final = percent_100_reached
        training_paths_total.extend(training_paths)
        print(training_paths)
        apis_empty = []
        # if there are no more apis left to find, the conversaiton is over!
        for k,v in agent.conversation_state.apis_for_eval.items():
            apis_empty.append(len(v['success'])==0)
        if all(apis_empty):
            print('Success reached 100%!!!!')
            break
        count+=1
        # print(agent_response)
    with open(f"{cwd}/records/llama_sims/sim_{convo_key.split('.json')[0]}.json", 'w') as file:
        json.dump({'internal_messages':conversations_total, 'training_paths': training_paths_total, '100_percent_reached':percent_100_reached_final}, file, indent=2)

    conversation_state.close_convos()
    conversation_state = None
    del conversation_state
    agent = None
    del agent
    user = None
    del user
