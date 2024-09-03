from multiwoz_api.utils import *
import openai
from openai import OpenAI
from user_simulator import UserSimulator
from agent_simulator_pref_tree_v2 import AgentSimulator
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

from peft import PeftModel, PeftConfig
model = PeftModel.from_pretrained(model, f'{cwd}/checkpoints/kto-works-lambdaD=1.7_checkpoint-240')
model = model.merge_and_unload()

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


path = f'{cwd}/records/training_data_v2/'
round_1_train = set([x.split('sim_')[-1] for x in os.listdir(path)])

already_done = ['MUL0649.json', 'MUL0652.json', 'MUL0653.json', 'MUL0655.json', 'MUL0656.json', 'MUL0658.json', 'MUL0659.json', 'MUL0660.json', 'MUL0661.json', 'MUL0662.json', 'MUL0664.json', 'MUL0665.json', 'MUL0666.json', 'MUL0667.json', 'MUL0668.json', 'MUL0670.json', 'MUL0672.json', 'MUL0678.json', 'MUL0679.json', 'MUL0680.json', 'MUL0683.json', 'MUL0684.json', 'MUL0685.json', 'MUL0686.json', 'MUL0688.json', 'MUL0689.json', 'MUL0692.json', 'MUL0693.json', 'MUL0695.json', 'MUL0697.json', 'MUL0698.json', 'MUL0699.json', 'MUL0700.json', 'MUL0701.json', 'MUL0704.json', 'MUL0705.json', 'MUL0708.json', 'MUL0709.json', 'MUL0710.json', 'MUL0711.json', 'MUL0712.json', 'MUL0714.json', 'MUL0715.json', 'MUL0716.json', 'MUL0717.json', 'MUL0718.json', 'MUL0720.json', 'MUL0721.json', 'MUL0722.json', 'MUL0723.json', 'MUL0724.json', 'MUL0725.json', 'MUL0726.json', 'MUL0727.json', 'MUL0728.json', 'MUL0730.json', 'MUL0732.json', 'MUL0733.json', 'MUL0734.json', 'MUL0735.json', 'MUL0736.json', 'MUL0740.json', 'MUL0741.json', 'MUL0742.json', 'MUL0743.json', 'MUL0746.json', 'MUL0747.json', 'MUL0748.json', 'MUL0749.json', 'MUL0750.json', 'MUL0752.json', 'MUL0753.json', 'MUL0755.json', 'MUL0756.json', 'MUL0757.json', 'MUL0758.json', 'MUL0762.json', 'MUL0763.json', 'MUL0764.json', 'MUL0765.json', 'MUL0766.json', 'MUL0767.json', 'MUL0768.json', 'MUL0769.json', 'MUL0770.json', 'MUL0771.json', 'MUL0774.json', 'MUL0775.json', 'MUL0776.json', 'MUL0777.json', 'MUL0778.json', 'MUL0779.json', 'MUL0780.json', 'MUL0781.json', 'MUL0782.json', 'MUL0783.json', 'MUL0784.json', 'MUL0786.json', 'MUL0788.json', 'MUL0790.json', 'MUL0792.json', 'MUL0793.json', 'MUL0794.json', 'MUL0795.json', 'MUL0796.json', 'MUL0797.json', 'MUL0799.json', 'MUL0800.json', 'MUL0802.json', 'MUL0804.json', 'MUL0809.json', 'MUL0811.json', 'MUL0817.json', 'MUL0820.json', 'MUL0824.json', 'MUL0825.json', 'MUL0833.json', 'MUL0835.json', 'MUL0836.json', 'MUL0840.json', 'MUL0847.json', 'MUL0853.json', 'MUL0854.json', 'MUL0855.json', 'MUL0862.json', 'MUL0867.json', 'MUL0868.json', 'MUL0877.json', 'MUL0879.json', 'MUL0885.json', 'MUL0886.json', 'MUL0887.json', 'MUL0892.json', 'MUL0895.json', 'MUL0898.json', 'MUL0900.json', 'MUL0905.json', 'MUL0909.json', 'MUL0911.json', 'MUL0913.json', 'MUL0918.json', 'MUL0919.json', 'MUL0920.json', 'MUL0921.json', 'MUL0922.json', 'MUL0925.json', 'MUL0927.json', 'MUL0928.json', 'MUL0931.json', 'MUL0935.json', 'MUL0936.json', 'MUL0940.json', 'MUL0944.json', 'MUL0946.json', 'MUL0948.json', 'MUL0950.json', 'MUL0952.json', 'MUL0955.json', 'MUL0956.json', 'MUL0961.json', 'MUL0973.json', 'MUL0975.json', 'MUL0976.json', 'MUL0977.json', 'MUL0981.json', 'MUL0982.json', 'MUL0983.json', 'MUL0984.json', 'MUL0986.json', 'MUL0987.json', 'MUL0988.json', 'MUL0989.json', 'MUL0991.json', 'MUL0993.json', 'MUL0997.json', 'MUL0998.json', 'MUL0999.json', 'MUL1004.json', 'MUL1006.json', 'MUL1007.json', 'MUL1009.json', 'MUL1010.json', 'MUL1011.json', 'MUL1013.json', 'MUL1016.json', 'MUL1017.json', 'MUL1019.json', 'MUL1020.json', 'MUL1021.json', 'MUL1023.json', 'MUL1026.json', 'MUL1029.json', 'MUL1030.json', 'MUL1032.json', 'MUL1036.json', 'MUL1037.json', 'MUL1039.json', 'MUL1040.json', 'MUL1041.json', 'MUL1047.json', 'MUL1048.json', 'MUL1052.json', 'MUL1053.json', 'MUL1056.json', 'MUL1063.json', 'MUL1065.json', 'MUL1067.json', 'MUL1068.json', 'MUL1070.json', 'MUL1074.json', 'MUL1075.json', 'MUL1079.json', 'MUL1081.json', 'MUL1084.json', 'MUL1085.json', 'MUL1089.json', 'MUL1092.json', 'MUL1095.json', 'MUL1103.json', 'MUL1104.json', 'MUL1106.json', 'MUL1108.json', 'MUL1111.json', 'MUL1112.json', 'MUL1113.json', 'MUL1115.json', 'MUL1118.json', 'MUL1120.json', 'MUL1121.json', 'MUL1123.json', 'MUL1124.json', 'MUL1129.json', 'MUL1136.json', 'MUL1144.json', 'MUL1145.json', 'MUL1147.json', 'MUL1149.json', 'MUL1150.json', 'MUL1152.json', 'MUL1153.json', 'MUL1157.json', 'MUL1164.json', 'MUL1165.json', 'MUL1168.json', 'MUL1170.json', 'MUL1171.json', 'MUL1173.json', 'MUL1178.json', 'MUL1179.json', 'MUL1180.json', 'MUL1187.json', 'MUL1188.json', 'MUL1191.json', 'MUL1195.json', 'MUL1197.json', 'MUL1198.json', 'MUL1199.json', 'MUL1203.json', 'MUL1204.json', 'MUL1205.json', 'MUL1206.json', 'MUL1208.json', 'MUL1209.json', 'MUL1214.json', 'MUL1216.json', 'MUL1218.json', 'MUL1223.json', 'MUL1225.json', 'MUL1230.json', 'MUL1231.json', 'MUL1234.json', 'MUL1241.json', 'MUL1251.json', 'MUL1257.json', 'MUL1259.json', 'MUL1260.json']

train_ids = list(set([k for k in apis.keys()]) - set(test_ids) - set(val_ids) - round_1_train - set(already_done))

import random
train_ids = sorted(train_ids)
random.Random(42).shuffle(train_ids)
set_to_run = train_ids[600:725]

results = []
failed_apis=[]
training_paths_total = []
to_save = []
record_folder_name = "kto_round_2_data"
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
    save_data = {'internal_messages':conversations_total, 'training_paths': training_paths_total, '100_percent_reached':percent_100_reached_final, 'user_cost':user_cost}
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
