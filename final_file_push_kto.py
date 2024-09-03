import json
import os
import pandas as pd
import yaml
import numpy as np
import os
import copy

path = os.getcwd()

with open(f'{path}/prompts/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)
with open(f'{path}/data/tools.json', 'r') as file:
    tools_list = json.load(file)
MONO_PROMPT = prompts['mistral_mono_prompt_v2'].replace('{example_filled}', json.dumps(tools_list, indent=2))



folder = f'{path}/records/training_data_v2/'
file_list = [folder+x for x in os.listdir(folder) if '.json' in x]
file_list = sorted(file_list, key=lambda x: x.split('.json')[0].split('_')[-1])
files = []
bad_ex = []
for fname in file_list:
    with open(fname, 'r') as file:
        tmp_data = json.load(file)
        if tmp_data['100_percent_reached']:
            new_bads = []
            longest = []
            for path, successful, is_gold in tmp_data['training_paths']:
                if is_gold and len(path) > len(longest):
                    longest = path
                if not successful and not is_gold:
                    new_bads.append(path)
            files.append(longest)
            bad_ex.append(new_bads)

def testing(utterance):
    if utterance['role']=='assistant':
        if 'APIRETURN' in utterance['content']:
            return False
    if 'ERROR:' in utterance['content']:
        return False
    if 'Error:' in utterance['content']:
        return False
    if 'FAILURE' in utterance['content']:
        return False
    return True
tots = []
bads = []
for j, f in enumerate(files):
    if all([testing(q) for q in f]):
        tots.append(f)
        for ex in bad_ex[j]:
            bads.append(ex)

files = tots



kto_data_dic_full_convo_only = {'data':[]}
unique_prompts = []
# add good
for jj, training_example in enumerate(files):
    if len(training_example)==0 or training_example in unique_prompts:
        continue
    unique_prompts.append(training_example)
    cleaned_training_example = [{'role':'system', 'content':MONO_PROMPT}]
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
    unique_prompts.append(training_example)
    cleaned_training_example = [{'role':'system', 'content':MONO_PROMPT}]
    for idx, dic in enumerate(reversed(training_example)):
        if dic.get('role')=='system':
            continue
        if dic.get('role')=='user' and 'APIRETURN' not in dic.get('content'):
            break
        if not dic.get('content').startswith('APIRETURN'):
            bad_data.append({'prompt':copy.deepcopy([{'role':'system', 'content':MONO_PROMPT}]+training_example[:len(training_example)-idx-1]), 'completion':[copy.deepcopy(dic)], 'label':False})

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

with open('train_data_kto_just_div.jsonl', 'w') as file:
    for idx, item in enumerate(kto_data_dic_full_convo_only['data']):
        json_line = json.dumps(item)
        additional = ''
        if idx < len(kto_data_dic_full_convo_only['data'])-1:
            additional = '\n'
        file.write(json_line + additional)

