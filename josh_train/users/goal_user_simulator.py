import copy
import yaml
from josh_train.users.base_user_simulator import BaseUserSimulator
from josh_train.utils import compute_cost, request_openai
import josh_train.config as config
from josh_train.josh import BaseJOSHAgent

class GoalUserSimulator(BaseUserSimulator):
    def __init__(self, goals, modelname, debug =False):
        self.modelname = modelname
        self.goals = goals['message']
        with open('prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.prompt = prompts['user_simulator_prompt']
        self.opening_prompt = prompts['user_simulator_opening_message_prompt'].replace('{goals}', '\n'.join(self.goals))
        self.system_prompt = prompts['user_simulator_system_prompt'].replace('{goals}', '\n'.join(self.goals))
        self.cost = 0.0
        self.debug = debug

    def make_messages(self, messages):
        if len(messages)==0:
            return [{'role': 'system', 'content': self.opening_prompt}]
        mod_messages = [{'role':'user' if message['role']=='assistant' else 'assistant', 'content':message['content']} for message in messages]
        return [{'role': 'system', 'content': self.system_prompt}] + mod_messages +  [{'role': 'system', 'content':self.prompt}]
    
    def step(self, agent):
        messages = copy.deepcopy(agent.messages)
        user_sim_messages = self.make_messages(messages=messages)
        output = request_openai(user_sim_messages, self.modelname, config.client)
        self.cost += compute_cost(user_sim_messages+ [{'role': 'user', 'content': output}])

        if self.debug:
            print('#'*30)
            print(f'USER: {output}')
            print('#'*30)

        agent.messages.append({'role': 'user', 'content': output})
        agent.messages_internal.append({'role': 'user', 'content': output})
        
        end_convo = 'END_CONVERSATION' in output
        return agent, end_convo
