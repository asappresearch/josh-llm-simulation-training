import math
from typing import Optional
import yaml
import re
import copy
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from josh_train.utils import make_transcript, request_openai, parse_api_call, handle_api_calls
import os
import josh_train.config as config
from josh_train.josh import BaseJOSHAgent


class ReACTAgentSimulator(BaseJOSHAgent):
    def __init__(self, api_examples, api_defs, model_name:Optional[str]=None, temperature=0.0, debug = False):
        super().__init__()
        cwd = os.getcwd()
        with open(f'{cwd}/prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}
        with open(f'{cwd}/data/tools.json', 'r') as file:
            tools_list = json.load(file)
        self.MONO_PROMPT = prompts['react_prompt'].replace('{example_filled}', json.dumps(tools_list, indent=2))
        self.pattern = "(PLAN|APICALL|SPEAK)(.*?)(?=PLAN|APICALL|SPEAK|$)"
        self.model_name=model_name
        self.debug = debug
        self.temperature = temperature

    def parse_agent_message(self, output):
        commands  = re.findall(self.pattern , output , re.DOTALL)
        return commands
    
    
    def request(self, messages, model=None, tokenizer=None) -> str:
        if model and tokenizer:
            encoding = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
            prompt_len=len(encoding[0])
            with torch.no_grad():
                if math.isclose(self.temperature, 0.0, rel_tol=1e-6):
                    generated_ids = model.generate(encoding, max_new_tokens=256, do_sample=False)
                else:
                    generated_ids = model.generate(encoding, max_new_tokens=256, temperature=self.temperature, top_k=50, top_p=0.95)
            return tokenizer.batch_decode(generated_ids[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0]
        else:
            output = request_openai(messages, self.model_name, config.client, temperature=self.temperature)
            return output
    
    def handle_api(self, command, conversation_state):
        try:
            api_values = parse_api_call(command)
        except:
            return 'FAILURE INCORRECTLY FORMATTED APICALL', None
        if api_values['api_name'] not in self.apis_to_examples:
            return 'FAILURE INCORRECTLY FORMATTED APICALL', None
        returns = handle_api_calls(api_values['api_name'], api_values['api_args'], conversation_state=conversation_state)
        if type(returns)==list:
            called_api = {'name':api_values['api_name'], 'parameters': api_values['api_args'], 'returned': returns[0] if len(returns)>0 else returns}
        else:
            called_api = {'name':api_values['api_name'], 'parameters': api_values['api_args'], 'returned': returns}
        return returns, called_api
    
    def step(self, **kwargs):
        conversation_state = kwargs['env']
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']

        self.recent_actions = []
        count=0
        while count < 3:
            agent_messages = [{'role':'system', 'content':self.MONO_PROMPT}]+self.messages_internal
            turn = self.request(agent_messages, model, tokenizer)
            if self.debug:
                print(turn)
            parsed = self.parse_agent_message(turn.replace('<COMMAND_END>', '').strip().replace('\n','').replace('\\',''))
            if len(parsed)==0:
                self.messages_internal.append({'role':'assistant', 'content':'ERROR: NO COMMAND FOUND'})
            thought_string = ''
            for command_type, command in parsed:
                command_type = command_type.strip()
                command=command.strip()
                if command_type=='PLAN':
                    thought_string = 'PLAN '+command+' <COMMAND_END> '
                elif command_type == 'SPEAK':
                    self.messages_internal.append({'role':'assistant', 'content':thought_string+'SPEAK '+command+' <COMMAND_END>'})
                    self.messages.append({'role':'assistant', 'content':command})
                    return 
                elif command_type == 'APICALL':
                    command = command.strip().replace('\n','')
                    output, called_api = self.handle_api(command, conversation_state)
                    self.recent_actions.append(called_api)
                    if self.debug:
                        print(output)
                    # Add the api call
                    self.messages_internal.append({'role':'assistant', 'content':thought_string+'APICALL '+command+' <COMMAND_END>'})
                    # Add the return
                    self.messages_internal.append({'role':'user', 'content':'APIRETURN ' + json.dumps(output)})
                else:
                    self.messages_internal.append({'role':'assistant', 'content':'ERROR: INVALID COMMAND TYPE'})
            count+=1
        self.messages.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return
