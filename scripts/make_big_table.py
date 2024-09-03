import json
import os
import numpy as np
first_half="""
\\begin{table*}[h!]
    \\centering
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{l|c|c|c|c|c|c}
        \\toprule
        Agent & """
indent_size = "        "
second_half="""
        \\bottomrule
    \\end{tabular}}
    \\caption{Performance on test set of multiple agents.}
    \\label{tab:performance}
\\end{table*}
"""
columns = ['API Success', 'Avg \# of goals achieved', 'API Success (vs guide simulator)', 'Win rate (vs LLama3-8B)', 'Avg \# of turns', 'Naturalness']
rows = {
    "OpenAI Assistant 3.5":dict(zip(columns, ['--']*len(columns))),
    "OpenAI Assistant 4o":dict(zip(columns, ['--']*len(columns))),
    "Llama 3 8B":dict(zip(columns, ['--']*len(columns))),
    "Llama 3 8B + SFT":dict(zip(columns, ['--']*len(columns))),
    "Llama 3 8B + KTO":dict(zip(columns, ['--']*len(columns))),
    "Llama 3 8B + SFT + KTO":dict(zip(columns, ['--']*len(columns))),
    "Llama 3 8B no partial credit":dict(zip(columns, ['--']*len(columns))),
}

cwd = os.getcwd()
result_paths = {
    "OpenAI Assistant 3.5":f'{cwd}/records/test_run_35_1106/',
    "OpenAI Assistant 4o":'',
    "Llama 3 8B":'',
    "Llama 3 8B + SFT":'',
    "Llama 3 8B + KTO":'',
    "Llama 3 8B + SFT + KTO":'',
    "Llama 3 8B no partial credit":'',
}

assert set(rows.keys()) == set(result_paths.keys())

for k,path in result_paths.items():
    if not path or len(path)==0:
        continue
    results = []
    num_turns = []
    assert len(os.listdir(path))==805
    for fname in os.listdir(path):
        with open(path+fname, 'r') as file:
            data = json.load(file)
        results.append(data['result'])
        num_turns.append(len([x for x in data['clean_messages'] if x['role'] == 'user']))
    rows[k]['API Success'] = float(np.mean(results))
    rows[k]['Avg \# of turns'] = float(np.mean(results))
    

# Build String
table = ' & '.join(columns)+' \\\\\n'+ indent_size + '\\midrule\n'
count = 0
for k,data in rows.items():
    newline=''
    if len(rows.keys())-1 > count:
        newline= '\n'
    
    # stylize the values
    vals = [str(round(data[col], 4)) if type(data[col])==float else data[col]  for col in columns]

    # join the row
    table = table + indent_size + k + ' & ' + ' & '.join(vals) + ' \\\\' + newline

    # add a hline as row 2
    if count ==2:
        table = table  + indent_size + '\\hline\n'
    count+=1

final_table = first_half+table+second_half

with open('table.txt', 'w') as file:
    file.write(final_table)