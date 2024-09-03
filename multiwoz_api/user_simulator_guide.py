import yaml
from multiwoz_api.utils import make_transcript, request_openai

guide_prompt = """You are a coach giving tips to a user simulator trying to replicate a conversation as consistently as possible. The user simulator is in the middle of a conversation, give it advice on what to do in the next turn.
Consistency means that over multiple runs, the user simulator should behave in the exact same way, it is your job to try and help it stay on the same trajectory every run.

###### Grounding Goals and Conversation #########
Customer goals:
{goals}

The following is the source conversation the user simulator is trying to replicate:
{goal_convo}
###################################################

######## CURRENT (real) Conversation #######################
This is the CURRENT conversaiton the user simulator is having:
{current_convo}

Use your best judgement if the conversation is not going well, it's possible the agent is not good enough and you need to end the conversation. End the conversation by putting END_CONVERSATION after your quote.
Keep in mind the Customer goals all must be communicated in order to give the agent enough information to properly search and book.
It is critical you give consistent advice over multiple iterations of the same conversation. The best way to do that is to ground your response in the source conversation and providing quotes whenever possible.
Please write breif advice on what the user simulator should say in order to keep it consistent and aligned with the source conversation. Write this advice to the user simulatior, referring to it as "you". No yapping.:

Example: 
Advice:
The user should ...
Suggested quote:
"Hello, how can I help you?"

Advice:
The conversation should be ended
Suggested quote:
"Thanks, goodbye" END_CONVERSATION

Output:
"""

class UserSimulator:
    def __init__(self, goals, convo, modelname='gpt-4o-2024-05-13'):
        self.goals = goals['message']
        with open('prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.prompt = prompts['user_simulator_prompt']
        self.modelname = modelname
        self.opening_prompt = prompts['user_simulator_opening_message_prompt'].replace('{goals}', '\n'.join(self.goals))
        self.system_prompt = prompts['user_simulator_system_prompt'].replace('{goals}', '\n'.join(self.goals))
        self.convo = []
        for log in convo['log']:
            tag = 'agent: ' if log['metadata'] else 'customer: '
            self.convo.append(tag+log['text'].strip())
    def make_messages(self, messages, system=None, prompt=None):
        mod_messages = [{'role':'user' if message['role']=='assistant' else 'assistant', 'content':message['content']} for message in messages]
        return [{'role': 'system', 'content': self.system_prompt if not system else system}] + mod_messages +  [{'role': 'system', 'content':self.prompt if not prompt else prompt}]
    
    def step(self, messages, client):
        user_sim_messages = self.make_messages(messages=messages)
        transcript = make_transcript(user_sim_messages, {'assistant':'user', 'user':'agent'})
        filled_guide_prompt = guide_prompt.replace('{goals}', '\n'.join(self.goals)).replace('{goal_convo}', '\n'.join(self.convo)).replace('{current_convo}', transcript if len(transcript)!=0 else 'NOT YET STARTED')
        output_guide = request_openai([{'role': 'user', 'content': filled_guide_prompt}], self.modelname, client)
        print(f'GUIDE: {output_guide}')
        output = output_guide.split("Suggested quote:")[-1].replace('\n', '').replace('"', '')
        return [{'role': 'user', 'content': output}]
