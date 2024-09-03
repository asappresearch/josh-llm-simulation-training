import yaml
import re
import copy
import json
from multiwoz_api.utils import make_transcript, request_openai, parse_api_call, handle_api_calls

class AgentSimulator:
    def __init__(self, api_examples, api_defs, client):
        with open('prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.prompt = prompts['agent_simulator_prompt']
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}
        self.system_prompt = prompts['agent_simulator_system_prompt'].replace('{api_name_list}', str([x['name'] for x in api_examples]))
        self.api_caller_system = prompts['api_caller_system']
        self.api_caller_prompt = prompts['api_caller_prompt']
        self.client = client
        self.openai_model = 'gpt-4'
        self.pattern = "(THOUGHTS|APICALL|SPEAK)(.*?)(?=THOUGHTS|APICALL|SPEAK|$)"

    def make_messages(self, messages):
        return [{'role': 'system', 'content': self.system_prompt}] + messages +  [{'role': 'system', 'content':self.prompt}]
    
    def parse_agent_message(self, output):
        commands  = re.findall(self.pattern , output , re.DOTALL)
        return commands
    
    def step(self, messages, conversation_state):
        messages = copy.deepcopy(messages)
        count=0
        while count < 10:
            agent_sim_messages = self.make_messages(messages=messages)
            output = request_openai(agent_sim_messages, self.openai_model, self.client)
            commands = self.parse_agent_message(output=output)
            for command_type, command in commands:
                if command_type == 'APICALL':
                    transcript = make_transcript(messages)
                    command = command.strip().replace('\n','')
                    if command not in self.apis_to_examples:
                        messages.append({'role': 'assistant', 'content': f'APICALL {command}'})
                        messages.append({'role': 'assistant', 'content': f'FAILURE INCORRECTLY FORMATTED APICALL'})
                        return messages
                        # raise ValueError # not a valid api call
                    api_shell = copy.deepcopy(self.apis_to_examples[command])
                    api_shell['parameters'] = {k:'' for k in api_shell['parameters'].keys()}
                    api_prompt = self.api_caller_prompt.replace('{conversation}', transcript).replace('{example_filled}', str(self.apis_to_examples[command])).replace('{api_shell}', str(api_shell))
                    api_message = [{'role': 'system', 'content':self.api_caller_system}, {'role': 'user', 'content': api_prompt}]
                    translated_api = request_openai(api_message, 'gpt-3.5-turbo-1106', self.client, response_format={"type": "json_object"})
                    api_values = parse_api_call(translated_api)
                    returns = handle_api_calls(api_values['api_name'], api_values['api_args'], conversation_state=conversation_state)
                    messages.append({'role': 'assistant', 'content': f'APICALL {command}'})
                    messages.append({'role': 'assistant', 'content': f'{returns}'})
                    print('#'*30)
                    print(f'APICALL {translated_api}')
                    print('#'*30)
                    print('#'*30)
                    print(f'APIRETURN {returns}')
                    print('#'*30)
                elif command_type == 'THOUGHTS':
                    messages.append({'role': 'assistant', 'content': f'THOUGHTS {command}'})
                    print('#'*30)
                    print(f'THOUGHTS {command}')
                    print('#'*30)
                elif command_type == 'SPEAK':
                    print('#'*30)
                    print(f'SPEAK {command}')
                    print('#'*30)
                    return [{'role': 'assistant', 'content': command}]
                else:
                    raise ValueError
            count+=1

        return [{'role': 'assistant', 'content': output}]