# Copyright Sierra

import copy
import json
from typing import Dict, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from multiwoz_api.josh import JOSH
from tau_bench.agents.base import BaseAgent
from tau_bench.agents.utils import (
    message_to_action,
    message_to_dict,
    pretty_print_conversation,
)

client = None


def initialize_client(**kwargs):
    global client
    client = OpenAI(**kwargs)

class JOSHAgent():
        def __init__(self, messages, env):
            self.messages = messages
            self.env = env
            self.reward = 0
            self.done = False
            self.info = None
            self.recent_action = None

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(
    messages: List[Dict[str, str]],
    model: str,
    tools=None,
    tool_choice="auto",
    temperature: float = 0.0,
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
    )
    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        tool_call = message.tool_calls[0]
        json.loads(tool_call.function.arguments)
    return message, dict(response.usage)


prompt_price_per_million = {
    "gpt-4o": 5,
    "gpt-4-turbo": 10,
    "gpt-4-32k-0613": 60,
    "gpt-3.5-turbo": 0.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}
completion_price_per_million = {
    "gpt-4o": 15,
    "gpt-4-turbo": 30,
    "gpt-4-32k-0613": 120,
    "gpt-3.5-turbo": 1.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}


class JOSHGPTFunctionCallingAgent(BaseAgent):
    def __init__(self, tools, wiki, model: str = "gpt-4-turbo"):
        self.tools = tools
        self.wiki = wiki
        self.model = model
        self.temperature = 1.0
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.wiki}]
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
##################################################

    def call_agent(self, agent):
        """
        Returns the agent and whether it is ready to pass to the customer
        """

        message, usage = chat_completion_request(
            agent.messages,
            model=self.model,
            tools=self.tools,
            temperature=self.temperature,
        )

        for key, value in usage.items():
            self.usage[key].append(value)
        if isinstance(message, Exception) and "context_length_exceeded" in str(
            message
        ):
            print(message)
            agent.done = True
            return agent, None
        
        action = message_to_action(message)
        agent.recent_action = action
        if action["name"] == "respond":
            agent.messages.append({"role": "assistant", "content": message.content})
            return agent, True
        
        obs, reward, done, info = agent.env.step(action)


        toolCall = message.tool_calls[0]
        # agent.messages.append(message)
        agent.messages.append({
                'role':'assistant',
                'tool_calls' : [ {
                    'id'       : toolCall.id,
                    'type'     : 'function',
                    'function' : {
                        'name'      : toolCall.function.name,
                        'arguments' : toolCall.function.arguments
                    }
                } ]
            }
        )
        agent.messages.append(
            {
                "role": "tool",
                "tool_call_id": toolCall.id,
                "content": obs,
            }
        )

        agent.reward = reward
        agent.done = done
        agent.info = info
        return agent, False
    

    def call_user(self, agent):
        obs, reward, done, info = agent.env.step(agent.recent_action)
        agent.messages.append({"role": "user", "content": obs})
        agent.reward = reward
        agent.done = done
        agent.info = info
        return agent

    def add_error_message(self, agent):
        agent.messages.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        agent.recent_action = {"name": 'respond', "arguments": 'Error: Agent ran out of retries.'}
        return agent


    def act(self, env, index=None, verbose=False, temperature=0.0):
        self.reset()
        obs, info = env.reset(index=index)
        reward = 0
        self.messages.append({"role": "user", "content": obs})
        josh = JOSH(
                rewards=copy.deepcopy(env.task['actions']), 
                agent_step=self.call_agent,
                user_step=self.call_user, 
                add_error_message=self.add_error_message,
                root_agent = JOSHAgent(self.messages, env)
            )
        
        max_reward = 0.0
        for _ in range(15):
            try:
                max_reward, all_done = josh.step()
            except Exception as e:
                print(e)
                info["error"] = str(e)
                break
            if all_done:
                break
        if len(josh.rewards)==0:
            assert josh.golden_agent is not None
            reward, info = josh.golden_agent.env.calculate_reward()
            max_reward = reward

        training_examples = []
        for ex in josh.training_examples:
            example = ({'messages':ex[0]}, ex[1], ex[2])
            example[0]['tools'] = self.tools
            training_examples.append(example)

        return max_reward, {'training_examples':training_examples}


    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        return [message_to_dict(message) for message in self.messages]
