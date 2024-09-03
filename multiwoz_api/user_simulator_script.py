import yaml
from multiwoz_api.utils import make_transcript, request_openai

class UserSimulator:
    def __init__(self, goals, convo):
        self.convo = [x['text'] for x in convo['log'] if not x['metadata']]
        self.idx = -1
    def step(self, messages, client):
        self.idx += 1
        if self.idx >= len(self.convo):
            return [{'role':'user', 'content':'END_CONVERSATION'}]
        return [{'role': 'user', 'content': self.convo[self.idx]}]