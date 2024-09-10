import yaml
import copy
import json
from josh_train.utils import request_openai, handle_api_calls
import os
import josh_train.config as config
import josh_train.josh as BaseJOSHAgent

class FCAgentSimulator(BaseJOSHAgent):
    def __init__(self, api_examples, api_defs, model_name, debug=False):
        cwd = os.getcwd()
        with open(f'{cwd}/prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}

        if os.path.isfile('data/tools.json'):
            with open('data/tools.json', 'r') as file:
                tools_list = json.load(file)
        else:
            tools_list = self._create_oai_function_list(api_examples)
            with open('data/tools_updated.json', 'w') as file:
                json.dump(tools_list, file, indent=2)
        self.tool_list = tools_list
        self.MONO_PROMPT = "You are a travel agent. Help the customer with the provided apis. Do not say you can do something you cannot. You can only do things with the provided apis."
        self.messages_full = []
        self.modelname = model_name
        self.debug = debug

        self.done = False
        self.recent_action = None

    def _create_oai_function_list(self, api_examples):
        official_list = []
        for item in api_examples:
            action, domain = item['name'].split('_')
            api_def = {'type':'function', 'function':{'name':item['name'], 'description': f"Allows you to {action} a {domain}", 'parameters':{'type':'object', 'required':[], 'properties':{}}}}
            for param in item['parameters'].keys():
                api_def['function']['parameters']['properties'][param] = {'type': 'string', 'description':f'One example would be {item["parameters"][param]}'}
            official_list.append(api_def)
        return official_list
    
    def request(self, messages):
        output = request_openai(messages, self.modelname, config.client, tools=self.tool_list)
        return output
    

    def _step(self, turn, conversation_state):
        msg_return = []
        if turn.tool_calls is not None:
            # API Call
            tool_call = turn.tool_calls[0]
            msg_return.append({
                    'role':'assistant',
                    'tool_calls' : [ {
                        'id'       : tool_call.id,
                        'type'     : 'function',
                        'function' : {
                            'name'      : tool_call.function.name,
                            'arguments' : tool_call.function.arguments
                        }
                    } ]
                }
            )

            api_call = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            if self.debug:
                print(json.dumps(api_call))
            api_returns = handle_api_calls(api_call['name'], api_call['arguments'], conversation_state=conversation_state)
            if self.debug:
                print(json.dumps(api_returns))
            # Add the return
            msg_return.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(api_returns),
                    }
                )

            return msg_return
        else:
            # Speak
            if self.debug:
                print(turn.content)
            msg_return.append({'role':'assistant', 'content':turn.content})
            return msg_return
    
    def step(self, messages, conversation_state):
        self.messages_full.extend(messages)
        count=0
        while count < 3:
            agent_messages = [{'role':'system', 'content':self.MONO_PROMPT}]+self.messages_full
            turn = self.request(agent_messages)
            message_return = self._step(turn, conversation_state)
            self.messages_full.extend(message_return)
            if all([x['role']!='tool' for x in message_return]):
                return message_return
            count+=1
        # self.messages_full.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return [{'role':'assistant', 'content':'Error: Agent ran out of retries.'}]
        