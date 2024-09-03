from multiwoz_api.utils import *
import openai
from openai import OpenAI
from user_simulator_guide import UserSimulator
from oai_agent_simulator import AgentSimulator
from conversation_state import Conversation
import numpy as np
import os
import time
from tqdm import tqdm

cwd = os.getcwd()
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

with open('data/testListFile.json', 'r') as file:
    test_ids = file.readlines()
test_ids = [x.strip() for x in test_ids]
test_ids = list(set(test_ids).intersection(apis.keys()))
with open('data/valListFile.json', 'r') as file:
    val_ids = file.readlines()
val_ids = [x.strip() for x in val_ids]
val_ids = set(val_ids).intersection(apis.keys())

train_ids = list(set([k for k in apis.keys()]) - set(test_ids) - set(val_ids))

set_to_run = val_ids

results = []
failed_apis=[]

# files = set([x.split('_')[1]+'.json' for x in os.listdir('/Users/blattimer/code/multiwoz-api/records/official_openai_runs/val_run_guide_2/')])
# set_to_run = list(val_ids - files)
for idx, convo_key in enumerate(tqdm(set_to_run)):
    conversation_state = Conversation(convo_key, apis, delex)
    user = UserSimulator(goals=conversation_state.goals, convo = real_convos[convo_key])
    agent = AgentSimulator(client, 'gpt-3.5-turbo-1106')

    # Run the simulation
    messages = []
    messages_with_apis = []
    while len(messages)<15:
        user_response = user.step(messages, client)
        messages += user_response
        messages_with_apis += user_response
        print('#'*30)
        print(f'USER: {user_response[0]["content"]}')
        print('#'*30)
        if 'END_CONVERSATION' in user_response[0]['content']:
            break

        agent_response, full_response = agent.step(user_response, conversation_state)
        messages += agent_response
        messages_with_apis += full_response
    result, failed_api = conversation_state.evaluate_apis()
    print(result)
    print(failed_api)
    failed_apis.append(failed_api)
    results.append(result)
    conversation_state.close_convos()
    conversation_state = None
    del conversation_state
    with open(f"{cwd}/records/official_openai_runs/val_run_guide_2/simulation_{convo_key.split('.json')[0]}_{idx}_{time.ctime().replace(' ','_')}.json", 'w') as file:
        json.dump({'result':result, 'failed_apis':failed_api, 'messages':messages_with_apis, 'id':convo_key}, file, indent=2)
print(results)
print(failed_api)
print(np.mean(results))
