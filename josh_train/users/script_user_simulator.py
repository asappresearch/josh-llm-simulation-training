import yaml
from josh_train.utils import make_transcript, request_openai
from josh_train.users.base_user_simulator import BaseUserSimulator

class ScriptUserSimulator(BaseUserSimulator):
    def __init__(self, goals, convo):
        self.convo = [x['text'] for x in convo['log'] if not x['metadata']]
        self.idx = -1
    def step(self, messages):
        self.idx += 1
        if self.idx >= len(self.convo):
            return [{'role':'user', 'content':'END_CONVERSATION'}]
        return [{'role': 'user', 'content': self.convo[self.idx]}]