
from typing import Dict, List
from josh_train.josh import BaseJOSHAgent

class BaseUserSimulator:
    def __init__(self):
        pass
    def step(self, agent:BaseJOSHAgent):
        return agent, True