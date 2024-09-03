import json
import os
import yaml
import os

path = os.getcwd()

with open(f'{path}/prompts/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)
with open(f'{path}/data/tools.json', 'r') as file:
    tools_list = json.load(file)
MONO_PROMPT = prompts['mistral_mono_prompt_v2'].replace('{example_filled}', json.dumps(tools_list, indent=2))

folder = f'{path}/records/mini_training_data_full/'
file_list = [folder+x for x in os.listdir(folder) if '.json' in x]
file_list = sorted(file_list, key=lambda x: x.split('.json')[0].split('_')[-1])
files = []
for fname in file_list:
    with open(fname, 'r') as file:
        tmp_data = json.load(file)
        if tmp_data['100_percent_reached']:
            longest = []
            for path, _, is_gold in tmp_data['training_paths']:
                if is_gold and len(path) > len(longest):
                    longest = path
            files.append(longest)
print(len(files))
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
for f in files:
    if all([testing(q) for q in f]):
        tots.append(f)

files = tots

kto_data_dic_full_convo_only = {'data':[]}
for f in files:
    if not f or len(f) ==0:
        continue
    kto_data_dic_full_convo_only['data'].append({'messages':[{'role':'system', 'content':MONO_PROMPT}] + f})
print(len(kto_data_dic_full_convo_only['data']))
with open('training_data_mini_0807.json', 'w') as file:
    json.dump(kto_data_dic_full_convo_only, file, indent=2)

with open('train_data_mini_0807.jsonl', 'w') as file:
    for idx, item in enumerate(kto_data_dic_full_convo_only['data']):
        json_line = json.dumps(item)
        additional = ''
        if idx < len(kto_data_dic_full_convo_only['data'])-1:
            additional = '\n'
        file.write(json_line + additional)

