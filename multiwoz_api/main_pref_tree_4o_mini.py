from multiwoz_api.utils import *
import openai
from openai import OpenAI
from user_simulator import UserSimulator
from agent_simulator_pref_tree_4o_mini import AgentSimulator
from conversation_state_pref_tree import Conversation
import numpy as np
import os
import torch

cwd = os.getcwd()

torch.cuda.empty_cache()

creds = get_openai_creds()
openai.api_key = creds['openai_key']
openai.organization = creds['openai_org']
api_key = creds['openai_key']
api_org = creds['openai_org']
client = OpenAI(api_key=api_key, organization=api_org)


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


with open('training_fnames.json', 'r') as file:
    training_fnames = json.load(file)

train_ids = sorted(training_fnames)

set_to_run = train_ids

results = []
failed_apis=[]
training_paths_total = []
to_save = []
record_folder_name = "mini_training_data"
for idx, convo_key in enumerate(set_to_run):
    conversation_state = Conversation(convo_key, apis, delex)
    user = UserSimulator(goals=conversation_state.goals, modelname='gpt-4o-2024-05-13')
    agent = AgentSimulator(api_examples, valid_api_defs, modelname='gpt-4o-mini-2024-07-18', client=client, conversation_state=conversation_state)

    # Run the simulation
    conversations = []
    conversations_total = []
    training_paths_total = []
    percent_100_reached_final=False
    count = 0
    user_cost = 0
    while count<10:
        print(count)
        replies = []
        if len(conversations)==0:
            response, cost = user.step([], client)
            replies = [response]
            user_cost+=cost
        else:
            for messages in conversations:
                response, cost = user.step(messages, client)
                replies.append(response)
                user_cost+=cost

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
    new_fname = f"{cwd}/records/{record_folder_name}/sim_{convo_key.split('.json')[0]}.json"
    save_data = {'internal_messages':conversations_total, 'training_paths': training_paths_total, '100_percent_reached':percent_100_reached_final, 'user_cost':user_cost, 'agent_cost':agent.cost}
    to_save.append({'file':new_fname, 'data':save_data})
    if idx % 25 ==0 or idx+1 >= len(set_to_run):
        for data_save in to_save:
            with open(data_save['file'], 'w') as file:
                json.dump(data_save['data'], file, indent=2)
        to_save=[]

    conversation_state.close_convos()
    conversation_state = None
    del conversation_state
    agent = None
    del agent
    user = None
    del user
