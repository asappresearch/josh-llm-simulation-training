
from typing import Dict, List


class BaseUserSimulator:
    def __init__(self):
        pass
    def step(self, messages: List[Dict[str,str]]) -> List[Dict[str,str]]:
        return messages