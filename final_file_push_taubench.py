import json
import os
import pandas as pd
import yaml
import numpy as np
import os
import copy

path = os.getcwd()

toolwoz=True

folder = '/Users/blattimer/code/josh-llm-simulation-training/records/beam-size-16-gpt-4o-mini-react-toolowz.json'#f'records/josh_function_calling_toolwoz_gpt-4o.json'
with open(folder, 'r') as file:
    tmp_data = json.load(file)

def filter_errors(convo):
    updated_convo = []
    for idx, msg in enumerate(convo['messages']):
        error_next = False
        if idx+1 < len(convo):
            error_next = ('content' in convo['messages'][idx+1] and 'Error' in convo['messages'][idx+1]['content']) or ('content' in msg and 'Error' in msg['content'])
        if not error_next:
            updated_convo.append(msg)
    convo['messages'] = updated_convo
    return convo

def no_error(convo):
    for msg in convo['messages']:
        if 'content' in msg and 'Error' in msg['content']:
            return False
        if 'content' in msg and msg['content'] == 'API output: ':
            return False
    return True

def no_error_toolwoz(convo):
    for utterance in convo['messages']:
        if 'content' not in utterance:
            continue 
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

files = []
bad_ex = []
for convo in tmp_data:
    if convo['reward']==1:
        new_bads = []
        longest = []
        convo_to_it = convo['info']['training_examples'] if not toolwoz else convo['training_examples']
        for path, successful, is_gold in convo_to_it:
            if is_gold and len(path['messages']) > len(longest):
                longest = path
        if toolwoz:
            if no_error_toolwoz(longest):
                files.append(longest)
        else:
            if no_error(longest):
                files.append(longest)
print(len(files))

import random
random.seed(42)
random.shuffle(files)

with open('train_data_ft-gpt-4o-mini-react-beam-16-toolwoz.jsonl', 'w') as file:
    for idx, item in enumerate(files):
        json_line = json.dumps(item)
        additional = ''
        if idx < len(files)-1:
            additional = '\n'
        file.write(json_line + additional)

