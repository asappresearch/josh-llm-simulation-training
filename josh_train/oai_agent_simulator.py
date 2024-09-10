import yaml
import re
import copy
import json
import time
import os
from josh_train.utils import make_transcript, request_openai, parse_api_call, handle_api_calls

class AgentSimulator:
    def __init__(self, client, modelname, api_examples=None):
        self.api_examples = api_examples
        self.openai_model = modelname
        self.client = client
        self.cost = 0
        self.modelname = modelname

        if os.path.isfile('data/tools.json'):
            with open('data/tools.json', 'r') as file:
                tools_list = json.load(file)
        else:
            tools_list = self._create_oai_function_list(api_examples)
            with open('data/tools_updated.json', 'w') as file:
                json.dump(tools_list, file, indent=2)
        

        self.assistant = client.beta.assistants.create(
            name="Travel Agent",
            instructions="You are a travel agent. Help the customer with the provided apis. Do not say you can do something you cannot. You can only do things with the provided apis.",
            model=self.modelname,
            tools=tools_list,
        )

        self.thread = self.client.beta.threads.create()
    
    def _create_oai_function_list(self, api_examples):
        official_list = []
        for item in api_examples:
            action, domain = item['name'].split('_')
            api_def = {'type':'function', 'function':{'name':item['name'], 'description': f"Allows you to {action} a {domain}", 'parameters':{'type':'object', 'required':[], 'properties':{}}}}
            for param in item['parameters'].keys():
                api_def['function']['parameters']['properties'][param] = {'type': 'string', 'description':f'One example would be {item["parameters"][param]}'}
            official_list.append(api_def)
        return official_list


    def step(self, messages, conversation_state):
        full_response = []
        message = self.client.beta.threads.messages.create(thread_id=self.thread.id, role=messages[-1]['role'], content=messages[-1]['content'])
        run = self.client.beta.threads.runs.create(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id
        )
        run = self.client.beta.threads.runs.retrieve(
        thread_id=self.thread.id,
        run_id=run.id
        )
        while run.status != 'completed':
            if run.required_action:
                tool_outputs = []
                for api_call in run.required_action.submit_tool_outputs.tool_calls:
                    #call api
                    output = {'tool_call_id': api_call.id, 'output':''}
                    output['output'] = json.dumps(handle_api_calls(api_call.function.name, json.loads(api_call.function.arguments), conversation_state=conversation_state))
                    print('#'*30)
                    print(f'APICALL {api_call.function.name}')
                    print(api_call.function.arguments)
                    call = {'name':api_call.function.name, 'parameters':json.loads(api_call.function.arguments)}
                    full_response.append({'role': 'assistant', 'content': f'APICALL {json.dumps(call)}'})
                    print(f'RETURN {output["output"]}')
                    full_response.append({'role': 'assistant', 'content': f'APIRETURN {output["output"]}'})
                    tool_outputs.append(output)
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                    )
                
            run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=run.id
            )
            time.sleep(0.1)
            if run.status=='failed' or run.status=='cancelled':
                break
        
        m = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
            )
        self.cost += ((0.5/1000000)*run.usage.prompt_tokens + (1.5/1000000) * run.usage.completion_tokens)
        print(self.cost)
        response = ' '.join([msg.text.value for msg in m.data[0].content])
        print(f'SPEAK {response}')
        full_response.append({'role': 'assistant', 'content': f'SPEAK {response}'})
        print('#'*30)
        return [{'role': 'assistant', 'content': response}], full_response