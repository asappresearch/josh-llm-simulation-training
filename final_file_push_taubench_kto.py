import json
import os
import pandas as pd
import yaml
import numpy as np
import os
import copy

path = os.getcwd()



folder = f'/root/code/multiwoz-api/results/react1-llama-1.0_range_0--1_usergpt-4o_0819233641.json'
with open(folder, 'r') as file:
    tmp_data = json.load(file)

files = []
bad_ex = []
for convo in tmp_data:
    if convo['reward']==1:
        new_bads = []
        longest = []
        for path, successful, is_gold in convo['info']['training_examples']:
            if is_gold and len(path) > len(longest):
                longest = path
            if not successful and not is_gold:
                new_bads.append(path)
        files.append(longest)
        bad_ex.append(new_bads)
print(len(files))
# def testing(utterance):
#     if utterance['role']=='assistant':
#         if 'API output:' in utterance['content']:
#             return False
#     if 'ERROR:' in utterance['content']:
#         return False
#     if 'Error:' in utterance['content']:
#         return False
#     if 'FAILURE' in utterance['content']:
#         return False
#     return True
# tots = []
bads = []
for j, f in enumerate(files):
    for ex in bad_ex[j]:
        bads.append(ex)
# print(files[0])
# files = tots
# print(len(files))

kto_data_dic_full_convo_only = {'data':[]}
unique_prompts = []
# add good
for jj, training_example in enumerate(files):
    if len(training_example)==0 or training_example in unique_prompts:
        continue
    unique_prompts.append(training_example)
    cleaned_training_example = []
    for idx, utterance in enumerate(training_example):
        if utterance['role']=='assistant':
            kto_data_dic_full_convo_only['data'].append({'prompt':copy.deepcopy(cleaned_training_example), 'completion':[copy.deepcopy(utterance)], 'label':True})
            cleaned_training_example.append(utterance)
        else:
            cleaned_training_example.append(utterance)

# add bad
bad_data = []
for jj, training_example in enumerate(bads):
    if len(training_example)==0 or training_example in unique_prompts:
        continue
    if training_example[-1].get('role')=='user':
        training_example = training_example[:-1]
    unique_prompts.append(training_example)
    for idx, dic in enumerate(reversed(training_example)):
        if dic.get('role')=='system':
            continue
        if dic.get('role')=='user' and 'API output:' not in dic.get('content'):
            break
        if not dic.get('content').startswith('API output:'):
            bad_data.append({'prompt':copy.deepcopy(training_example[:len(training_example)-idx-1]), 'completion':[copy.deepcopy(dic)], 'label':False})

print(len(kto_data_dic_full_convo_only['data']))
shared_bads = []

for good_example in kto_data_dic_full_convo_only['data']:
    c=[]
    good_prompt = good_example['prompt']
    for bad_example in bad_data:
        if bad_example['prompt'] == good_prompt and bad_example not in shared_bads:
            shared_bads.append(bad_example)

print(len(shared_bads))
kto_data_dic_full_convo_only['data'] = kto_data_dic_full_convo_only['data']+shared_bads
# kto_data_dic_full_convo_only = {'data':[]}
# for f in files:
#     if not f or len(f) ==0:
#         continue
#     kto_data_dic_full_convo_only['data'].append({'messages':[{'role':'system', 'content':MONO_PROMPT}] + f})

with open('train_data_kto_retail.jsonl', 'w') as file:
    for idx, item in enumerate(kto_data_dic_full_convo_only['data']):
        json_line = json.dumps(item)
        additional = ''
        if idx < len(kto_data_dic_full_convo_only['data'])-1:
            additional = '\n'
        file.write(json_line + additional)

