import yaml
import re
import copy
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from josh_train.utils import make_transcript, request_openai, parse_api_call, handle_api_calls
import os
class AgentSimulator:
    def __init__(self, api_examples, api_defs, model, tokenizer):
        cwd = os.getcwd()
        with open(f'{cwd}/prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}
        with open(f'{cwd}/data/tools.json', 'r') as file:
            tools_list = json.load(file)
        self.MONO_PROMPT = prompts['mistral_mono_prompt_v2'].replace('{example_filled}', json.dumps(tools_list, indent=2))
        self.pattern = "(PLAN|APICALL|SPEAK)(.*?)(?=PLAN|APICALL|SPEAK|$)"
        self.model = model
        self.tokenizer = tokenizer
        self.before_message = '\nOutput:\nPLAN '
        self.messages_full = []

    def parse_agent_message(self, output):
        commands  = re.findall(self.pattern , output , re.DOTALL)
        return commands
    
    
    def request(self, messages):
        encoding = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
        prompt_len=len(encoding[0])
        with torch.no_grad():
            generated_ids = self.model.generate(encoding, max_new_tokens=256, do_sample=False)#, temperature=0.7, top_k=50, top_p=0.95)
        return self.tokenizer.batch_decode(generated_ids[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)
        

    def make_messages(self, messages):
        result =[{'role':'system', 'content':self.MONO_PROMPT}]
        for idx, message in enumerate(messages):
            result.append(message)
        result[-1]['content'] = result[-1]['content']#+self.before_message
        return result
    
    def handle_api(self, command, conversation_state):
        try:
            api_values = parse_api_call(command)
        except:
            return 'FAILURE INCORRECTLY FORMATTED APICALL'
        if api_values['api_name'] not in self.apis_to_examples:
            return 'FAILURE INCORRECTLY FORMATTED APICALL'
        returns = handle_api_calls(api_values['api_name'], api_values['api_args'], conversation_state=conversation_state)
        return returns
    
    def step(self, messages, conversation_state):
        self.messages_full.extend(messages)
        count=0
        while count < 3:
            agent_messages = self.make_messages(self.messages_full)
            turn = self.request(agent_messages)
            print(turn[0])
            parsed = self.parse_agent_message(turn[0].replace('<COMMAND_END>', '').strip().replace('\n','').replace('\\',''))
            if len(parsed)==0:
                self.messages_full.append({'role':'assistant', 'content':'ERROR: NO COMMAND FOUND'})
            print(parsed)
            thought_string = ''
            for command_type, command in parsed:
                command_type = command_type.strip()
                command=command.strip()
                if command_type=='PLAN':
                    thought_string = 'PLAN '+command+' <COMMAND_END> '
                    # self.messages_full.append({'role':'assistant', 'content':'PLAN '+command})
                elif command_type == 'SPEAK':
                    self.messages_full.append({'role':'assistant', 'content':thought_string+'SPEAK '+command+' <COMMAND_END>'})
                    return [{'role':'assistant', 'content':command}], []
                elif command_type == 'APICALL':
                    command = command.strip().replace('\n','')
                    output = self.handle_api(command, conversation_state)
                    print(output)
                    # Add the api call
                    self.messages_full.append({'role':'assistant', 'content':thought_string+'APICALL '+command+' <COMMAND_END>'})
                    # Add the return
                    self.messages_full.append({'role':'user', 'content':'APIRETURN ' + json.dumps(output)})
                else:
                    self.messages_full.append({'role':'assistant', 'content':'ERROR: INVALID COMMAND TYPE'})
            count+=1

        # self.messages_full.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return [{'role':'assistant', 'content':'Error: Agent ran out of retries.'}], []
        