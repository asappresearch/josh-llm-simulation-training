import yaml
from multiwoz_api.utils import compute_cost, request_openai

class UserSimulator:
    def __init__(self, goals, modelname):
        self.modelname = modelname
        self.goals = goals['message']
        with open('prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.prompt = prompts['user_simulator_prompt']
        self.opening_prompt = prompts['user_simulator_opening_message_prompt'].replace('{goals}', '\n'.join(self.goals))
        self.system_prompt = prompts['user_simulator_system_prompt'].replace('{goals}', '\n'.join(self.goals))

    def make_messages(self, messages):
        if len(messages)==0:
            return [{'role': 'system', 'content': self.opening_prompt}]
        mod_messages = [{'role':'user' if message['role']=='assistant' else 'assistant', 'content':message['content']} for message in messages]
        return [{'role': 'system', 'content': self.system_prompt}] + mod_messages +  [{'role': 'system', 'content':self.prompt}]
    
    def step(self, messages, client):
        user_sim_messages = self.make_messages(messages=messages)
        output = request_openai(user_sim_messages, self.modelname, client)
        return [{'role': 'user', 'content': output}], compute_cost(user_sim_messages+ [{'role': 'user', 'content': output}])