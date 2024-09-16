import sqlite3
import json
import ast
from typing import Any, List
from dotenv import load_dotenv
import os
import time
import tiktoken
import re
from colorama import Back, Fore, Style
import pandas as pd
import pandasql as psql

with open('data/valid_api_defs.json', 'r') as file:
    valid_api_defs = json.load(file)

valid_apis = {}
for k in valid_api_defs.keys():
    for api in valid_api_defs[k].keys():
        valid_apis[api] = valid_api_defs[k][api]

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"
HF_TOKEN = "HF_TOKEN"

def _get_env_val(key):
    val = os.environ.get(key, "").strip()
    return val or None

def printc(
    *args: Any, color: str = "", back: str = "", style: str = "", **kwargs: Any
) -> None:
    """
    Helper method for printing colored and styled strings.

    Args:
        color: colorama.Fore foreground color value
        back: colorama.Back background value
        style: colorama.Style style value
    """
    color_str = Fore.RESET + Style.RESET_ALL + Back.RESET + color + back + style
    print(color_str, end="")
    print(*args, **kwargs, end="")
    print(Fore.RESET + Style.RESET_ALL + Back.RESET)


def print_error(*args: Any, **kwargs: Any) -> None:
    printc(*args, **kwargs, color=Fore.RED)


def print_system(*args: Any, color: str = "", **kwargs: Any) -> None:
    printc(*args, **kwargs, color=color, style=Style.DIM)

def get_openai_creds():
    """
    Calls `load_dotenv` to load any environments variables from a `.env` file
    and then reads the OpenAI credentials from environment variables.

    Returns `OpenAICreds` if successful. Returns `None` if a required value
    is not found.

    If the creds are not found, this method will print an error indicating
    which environment variables were not found.

    Expected Environment Variables:
    - `OPENAI_API_KEY`
    - `OPENAI_ORGANIZATION`
    """
    load_dotenv()

    openai_key = _get_env_val(OPENAI_API_KEY)
    openai_org = _get_env_val(OPENAI_ORGANIZATION)

    missing_keys = []
    if not openai_key:
        missing_keys.append(OPENAI_API_KEY)
    if not openai_org:
        missing_keys.append(OPENAI_ORGANIZATION)

    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        print_error(
            f"Failed to load OpenAICreds. Missing envionment variable(s): {missing_keys_str}"
        )
        print_system(
            "Pro tip: you can create a `.env` file in your project's root directory with these "
            + "values, and they will be loaded automatically."
        )
        raise ValueError(f"Failed to load OpenAICreds. Missing envionment variable(s): {missing_keys_str}")

    return {
        'openai_key':openai_key,
        'openai_org':openai_org,
        }

def get_hf_creds():
    """
        Same as get_openai_creds above but for huggingface
    """
    load_dotenv()

    hf_token = _get_env_val(HF_TOKEN)

    missing_keys = []
    if not hf_token:
        missing_keys.append(HF_TOKEN)

    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        print_error(
            f"Failed to load HF Creds. Missing envionment variable(s): {missing_keys_str}"
        )
        print_system(
            "Pro tip: you can create a `.env` file in your project's root directory with these "
            + "values, and they will be loaded automatically."
        )
        raise ValueError(f"Failed to load HF Creds. Missing envionment variable(s): {missing_keys_str}")

    return {
        'hf_token':hf_token,
        }


def request_openai(messages, model, client, max_tokens=256, temperature=0.0, response_format=None, tools = None):
    retries = 0
    retry_limit = 4
    success = False
    while not success and retries < retry_limit:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=42,
                temperature=temperature,
                tools=tools)
            success = True
            break
        except:
            if retries +1 == retry_limit:
                time.sleep(30)
            else:
                time.sleep(10)
            retries+=1
    if not success:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format,
            seed=42,
            temperature=temperature,
            tools=tools)
        
    if tools is None:  
        return response.choices[0].message.content
    else:
        return response.choices[0].message

def compute_cost(messages, model='gpt-4o'):
    cost_dict = {'gpt-4o':(5/(10**6), 15/(10**6)),
                 'gpt-4o-2024-05-13':(5/(10**6), 15/(10**6)),
                 'gpt-4o-mini':(0.15/(10**6), 0.6/(10**6)),
                 'gpt-4o-mini-2024-07-18':(0.15/(10**6), 0.6/(10**6))}
    if model not in cost_dict:
        return 0.0
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    num_output_tokens = len(encoding.encode(messages[-1]['content']))
    cost = (num_tokens*cost_dict[model][0]) + (num_output_tokens*cost_dict[model][1])
    return cost

def create_dbs():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
    dbs = {}
    for domain in domains:
        db = 'db/{}_db.json'.format(domain)
        with open(db, 'r') as file:
            database = pd.DataFrame(json.load(file))
        dbs[domain] = database
    return dbs

def execute_query(dbs, domain, sql_query):
    result = psql.sqldf(sql_query, env=dbs)
    return result

def filter_dbs(domain, failure_cases, dbs):
    sql_query = "delete from {}".format(domain)

    flag = True
    tmp_df = dbs[domain].copy()
    for key, val in failure_cases.items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            val2 = val.replace("'", "''")
            # change query for trains
            if key == 'leaveAt':
                tmp_df = tmp_df[tmp_df[key]<=val2]
                # sql_query += r" " + key + " > " + r"'" + val2 + r"'"
            elif key == 'arriveBy':
                tmp_df = tmp_df[tmp_df[key]>=val2]
                # sql_query += r" " + key + " < " + r"'" + val2 + r"'"
            else:
                tmp_df = tmp_df[tmp_df[key]==val2]
                # sql_query += r" " + key + "=" + r"'" + val2 + r"'"
    # rval = execute_query(dbs, domain, sql_query)
    return tmp_df

def test_if_val_is_empty(val):
    return val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care"

def queryDataBase(domain, turn, dbs):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    tmp_df = dbs[domain].copy()
    for key, val in turn.items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care" or not val or type(val)!= str:
            pass
        else:
            val2 = val.replace("'", "''")
            # change query for trains
            if key == 'leaveAt':
                tmp_df = tmp_df[tmp_df[key]>val2]
                # sql_query += r" " + key + " > " + r"'" + val2 + r"'"
            elif key == 'arriveBy':
                tmp_df = tmp_df[tmp_df[key]<val2]
                # sql_query += r" " + key + " < " + r"'" + val2 + r"'"
            else:
                tmp_df = tmp_df[tmp_df[key]==val2]
                # sql_query += r" " + key + "=" + r"'" + val2 + r"'"

    # db_results = execute_query(dbs, domain, sql_query)
    return_results = [x.to_dict() for _, x in tmp_df.iterrows()]
    return return_results

def extract_and_parse_json(s):
    # Regex pattern to extract JSON dictionary
    json_pattern = r'\{.*\}'
    
    # Search for the JSON pattern in the string
    match = re.search(json_pattern, s)
    
    if match:
        json_str = match.group(0)
        
        try:
            # Parse the JSON string into a dictionary
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            print("Error: The JSON data could not be decoded.")
            return None
    else:
        print("Error: No JSON dictionary found in the string.")
        return None

def parse_api_call(api_call):
    """
    Turn an api call string into an api_name and api_args
    Example:
    {'name': 'get_coffee', 'parameters':{'type':'black', 'donut':True, 'number_of_coffees':10}}
    """
    api_values = {}
    api_compiled = extract_and_parse_json(api_call)
    api_values['api_name'] = api_compiled['name']
    api_values['api_args'] = api_compiled['parameters']
    return api_values


def validate_api_call(api_name, api_args):
    """
    Returns True or False whether the api call is correct with respect to the allowed apis
    """
    if api_name not in valid_apis:
        return False
    return (set(api_args.keys()).issubset(set(valid_apis[api_name]['parameters'])) )

def _serve_search_result_calls(domain, api_args, conversation_state, results):
    goal_parameters = conversation_state.apis[domain]['success'][f'search_{domain}']['parameters']
    booking_in_goal_calls = f'book_{domain}' in conversation_state.apis[domain]['success']
    if booking_in_goal_calls:
        book_id = conversation_state.apis[domain]['success'][f'book_{domain}']['unique_id']
        unique_id_type = 'trainID' if domain == 'train' else 'name'

    correct_answer = None
    wrong_answer = None
    for result in results:
        if booking_in_goal_calls and result[unique_id_type] == book_id:
            correct_answer = [result]
        if not is_subset(goal_parameters, result):
            wrong_answer = [result]

    # If the goal is a subset of the arguments
    if is_subset(goal_parameters, api_args):
        # if there is a technically "correct" answer to serve
        if booking_in_goal_calls:
            if correct_answer:
                return correct_answer
            else:
                if wrong_answer:
                    return wrong_answer
                else:
                    return []

    # If the arguments is a subset of the goal
    elif is_subset(api_args, goal_parameters):
        if wrong_answer:
            return wrong_answer
        elif booking_in_goal_calls and correct_answer:
            return correct_answer

    return [results[0]]
    

def _handle_api_calls(api_name, api_args, conversation_state):
    """
    Executes a given api call on the data
    """
    if not validate_api_call(api_name, api_args):
        return 'ERROR: INVALID API CALL! YOUR REQUEST CONTAINED PARAMETERS THAT ARE NOT ALLOWED FOR THIS API or AN IMPROPER API NAME!'
    for  k,v in api_args.items():
        if type(v)==list or type(v)==dict:
            return 'ERROR: INVALID API CALL! LISTS AND DICTS ARE NOT ALLOWED AS VALUES!'
        # raise ValueError
    action, domain = api_name.split('_')
    api_args = {k:v.lower().replace("'","") if str == type(v) else v for k,v in api_args.items()}
    if action == 'search':
        results = queryDataBase(domain, api_args, conversation_state.dbs)
        if len(results)==0:
            return results
        if domain not in conversation_state.apis:
            print('out of scope api')
            return [results[0]]
        # Make sure that if there's supposed to be a successful booking, we recommend the item for that booking if the search query is correct
        return _serve_search_result_calls(domain, api_args, conversation_state, results)
    elif action == 'book':
        if domain not in conversation_state.apis or f'book_{domain}' not in conversation_state.apis[domain]['success']:
            return {'success': False, 'return':{}}
        if is_subset(conversation_state.apis[domain]['success'][f'book_{domain}']['parameters'], api_args):
            return {'success': True, 'return':conversation_state.apis[domain]['success'][f'book_{domain}']['returns']}
        else:
            return {'success': False, 'return':{}}
    else:
        return 'ERROR: INVALID API CALL!'
        # raise ValueError

def handle_api_calls(api_name, api_args, conversation_state):
    """
    Executes a given api call on the data
    """
    return_value = _handle_api_calls(api_name, api_args, conversation_state)
    action, domain = api_name.split('_')
    unique_id_type = 'trainID' if domain == 'train' else 'name'
    #if the call was invalid
    if type(return_value) == str:
        unique_id = ''
    elif action == 'search':
        unique_id = return_value[0][unique_id_type] if len(return_value)>0 else ''
    elif action == 'book':
        unique_id = return_value['return'][unique_id_type] if unique_id_type in return_value['return'] else ''
    else:
        unique_id = ''

    conversation_state.add_api_call(api_name, api_args, return_value)
    return return_value


def make_transcript(messages, role_map=None):
    """
    Output a transcript of messages
    """
    transcript = ""
    for message in messages:
        if message['role']=='system':
            continue
        if role_map:
            transcript = transcript + role_map[message['role']] + ': ' + message['content'] + '\n'
        else:
            transcript = transcript + message['role'] + ': ' + message['content'] + '\n'
    return transcript

def create_apis(dbs):
    with open('data/data.json') as fin1:
        data = json.load(fin1)

    with open('data/valid_api_defs.json', 'r') as file:
        apis = json.load(file)

    with open('data/delex.json') as outfile:
        delex = json.load(outfile)

    searchable_domains = ['hotel', 'attraction', 'train', 'restaurant']

    convo_keys = [x for x in data.keys() if not data[x]['goal']['taxi'] and not data[x]['goal']['police'] and not data[x]['goal']['hospital']]
    convo_api_calls = {}
    for convo in convo_keys:
        final_api_calls = {}
        for domain in ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']:
            if data[convo]['goal'][domain]:
                # print(domain)
                # print(data[convo]['goal'][domain])
                final_api_calls[domain] = {'success': {}, 'failure': {}, 'requests':[]}
                ###### switch for taxi ##########
                if domain in ['taxi', 'hospital']:
                    assert 'book' not in data[convo]['goal'][domain]
                    data[convo]['goal'][domain]['book'] = data[convo]['goal'][domain]['info']
                    data[convo]['goal'][domain]['fail_book'] = data[convo]['goal'][domain]['fail_info']
                    data[convo]['goal'][domain]['info'] = {}
                    data[convo]['goal'][domain]['fail_info'] = {}
                #################### info/search ####################
                if 'info' in data[convo]['goal'][domain] and data[convo]['goal'][domain]['info']:
                    # Construct the final search call
                    parameters = {}
                    for api_param in apis[domain][f'search_{domain}']['parameters']:
                        parameters[api_param] = data[convo]['goal'][domain]['info'][api_param] if api_param in data[convo]['goal'][domain]['info'] else ''
                    
                    returns = queryDataBase(domain, parameters, dbs)
                    dict_returns = [dict(zip(apis[domain][f'search_{domain}']['returns'], returned_db_item.values())) for returned_db_item in returns]
                    final_api_calls[domain]['success'][f'search_{domain}'] = {'parameters':parameters, 'returns': dict_returns}

                if 'fail_info' in data[convo]['goal'][domain] and data[convo]['goal'][domain]['fail_info']:
                    # Construct the failed search call
                    parameters = {}
                    for api_param in apis[domain][f'search_{domain}']['parameters']:
                        if api_param in data[convo]['goal'][domain]['fail_info']:
                            parameters[api_param] = data[convo]['goal'][domain]['fail_info'][api_param]
                        else:
                            parameters[api_param] = data[convo]['goal'][domain]['info'][api_param] if api_param in data[convo]['goal'][domain]['info'] else ''
                    
                    returns = queryDataBase(domain, parameters, dbs)
                    dict_returns = [dict(zip(apis[domain][f'search_{domain}']['returns'], returned_db_item.values())) for returned_db_item in dict_returns]
                    final_api_calls[domain]['failure'][f'search_{domain}'] = {'parameters':parameters, 'returns': returns}
                #################### requests ####################
                if 'reqt' in data[convo]['goal'][domain]:
                    # we can zip these up with the successful results later to find their correct values
                    final_api_calls[domain]['requests'] = data[convo]['goal'][domain]['reqt']
                #################### booking ####################
                if 'book' in data[convo]['goal'][domain] and data[convo]['goal'][domain]['book']:
                    # Get the params
                    parameters = {}
                    for api_param in apis[domain][f'book_{domain}']['parameters']:
                        parameters[api_param] = data[convo]['goal'][domain]['book'][api_param] if api_param in data[convo]['goal'][domain]['book'] else ''
                    # Get the returns
                    last_sys_index = -1 if delex[convo]['log'][-1]['metadata'] else -2
                    assert delex[convo]['log'][last_sys_index]['metadata']
                    returns = delex[convo]['log'][last_sys_index]['metadata'][domain]['book']['booked']
                    if domain in searchable_domains:
                        assert f'search_{domain}' in final_api_calls[domain]['success']
                        #fill in missing keys
                        for missing_item_idx, return_item in enumerate(returns):
                            if set([x for x in return_item.keys()]) != set([x for x in apis[domain][f'book_{domain}']['returns']]):
                                for missing_key in set([x for x in apis[domain][f'book_{domain}']['returns']])-set([x for x in return_item.keys()]):
                                    if missing_key in delex[convo]['log'][last_sys_index]['metadata'][domain]['semi'] and not test_if_val_is_empty(delex[convo]['log'][last_sys_index]['metadata'][domain]['semi'][missing_key]):
                                        returns[missing_item_idx][missing_key] = delex[convo]['log'][last_sys_index]['metadata'][domain]['semi'][missing_key]
                                    else:
                                        returns[missing_item_idx][missing_key] = final_api_calls[domain]['success'][f'search_{domain}']['returns'][0][missing_key]
                    else:
                        unique_id=None
                    if len(returns)!=0:
                        # choose the final booking option from the state
                        returns = returns[-1]
                        unique_id_key = 'trainID' if domain == 'train' else 'name'
                        parameters[unique_id_key] = returns[unique_id_key]
                        final_api_calls[domain]['success'][f'book_{domain}'] = {'parameters':parameters, 'returns': returns, 'unique_id':returns[unique_id_key]}
                if 'fail_book' in data[convo]['goal'][domain] and data[convo]['goal'][domain]['fail_book']:
                    # Construct the failed search call
                    parameters = {}
                    for api_param in apis[domain][f'book_{domain}']['parameters']:
                        if api_param in data[convo]['goal'][domain]['fail_book']:
                            parameters[api_param] = data[convo]['goal'][domain]['fail_book'][api_param]
                        else:
                            parameters[api_param] = data[convo]['goal'][domain]['book'][api_param] if api_param in data[convo]['goal'][domain]['book'] else ''
                    parameters['trainID' if domain=='train' else 'name']=''
                    returns = []
                    final_api_calls[domain]['failure'][f'book_{domain}'] = {'parameters':parameters, 'returns': returns}
        convo_api_calls[convo] = final_api_calls
    return convo_api_calls

def create_api_examples(apis):
    examples = {}
    for valid_api in valid_apis.keys():
        if valid_api.split('_')[1].strip() not in ['train', 'restaurant', 'hotel', 'attraction']:
            continue
        examples[valid_api] = {k:'' for k in valid_apis[valid_api]['parameters']}
        domain = valid_api.split('_')[1].strip()
        for key in apis.keys():
            if domain in apis[key] and valid_api in apis[key][domain]['success']:
                for parameter in apis[key][domain]['success'][valid_api]['parameters'].keys():
                    if not test_if_val_is_empty(apis[key][domain]['success'][valid_api]['parameters'][parameter]):
                        examples[valid_api][parameter] = apis[key][domain]['success'][valid_api]['parameters'][parameter]

    return [{'name':k, 'parameters': examples[k]} for k in examples.keys()]


def is_subset(dict_subset, dict_main):
    cleaned_main = {k:v.lower().replace("'","") if type(v)==str else v for k,v in dict_main.items() if not test_if_val_is_empty(v)}
    cleaned_subset = {k:v.lower().replace("'","") if type(v)==str else v for k,v in dict_subset.items() if not test_if_val_is_empty(v)}
    return all(item in cleaned_main.items() for item in cleaned_subset.items())
