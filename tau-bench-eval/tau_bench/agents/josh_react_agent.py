# Copyright Sierra

import copy
import json
import time
import torch
from josh_train.josh import JOSH, BaseJOSHAgent, BaseRewards
# from tenacity import retry, stop_after_attempt, wait_random_exponential
from josh_train.utils import get_openai_creds
from openai import OpenAI

from tau_bench.agents.base import BaseAgent
from tau_bench.agents.utils import pretty_print_conversation

create = None
create_mode = None

def request_llama(messages, tokenizer, model, temperature):
    encoding = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
    prompt_len=len(encoding[0])+5
    with torch.no_grad():
        generated_ids = model.generate(encoding, max_new_tokens=256, do_sample=False)#, temperature=0.7, top_k=50, top_p=0.95)
    o = tokenizer.batch_decode(generated_ids[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0]
    return o

def initialize_create(mode="openai", **kwargs):
    global create, create_mode
    if mode == "openai":
        # def initialize_client(**kwargs):
        creds = get_openai_creds()
        create = OpenAI(api_key=creds['openai_key'], organization=creds['openai_org'], **kwargs).chat.completions.create
        create_mode = "openai"

    elif mode == "llama":
        create = request_llama
        create_mode = "llama"

    elif mode == "anthropic":
        from anthropic import Anthropic

        create = Anthropic().messages.create
        create_mode = "anthropic"

    elif mode == "google":
        global GenerativeModel
        from google.generativeai import GenerativeModel

        create = None
        create_mode = "google"




# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def get_message_action(
    messages, model, **kwargs
):  # kwargs only contain temperature for now
    global create, create_mode
    if create_mode == "openai":
        kwargs["model"] = model
        kwargs["messages"] = messages
    elif create_mode == "llama":
        kwargs["model"] = model
        kwargs["messages"] = messages
    elif create_mode == "anthropic":
        kwargs["system"] = messages[0]["content"]
        kwargs["max_tokens"] = 256
        kwargs["model"] = model
        kwargs["messages"] = messages[1:]
    elif create_mode == "google":
        create = GenerativeModel(
            model, system_instruction=messages[0]["content"], generation_config=kwargs
        ).generate_content
        kwargs = {
            "contents": [
                {
                    "role": {"user": "user", "assistant": "model"}[m["role"]],
                    "parts": [m["content"]],
                }
                for m in messages[1:]
            ]
        }
        time.sleep(2)

    response = create(**kwargs)

    if create_mode == "openai":
        message = response.choices[0].message.content
    elif create_mode == "llama":
        message = response
    elif create_mode == "anthropic":
        message = response.content[0].text
    elif create_mode == "google":
        message = response.text

    action_name = message.split("Action:")[-1].split("Arguments:")[0].strip()
    action_args = message.split("Arguments:")[-1].strip().split("\n")[0]
    if action_name == "respond" or action_name == "":
        action_args = {"content": action_args}
    else:
        action_args = json.loads(action_args)
    return message, {"name": action_name, "arguments": action_args}

class JOSHAgent(BaseJOSHAgent):
    def __init__(self, messages, env):
        super().__init__(messages=messages)
        self.env = env
        self.reward = 0
        self.done = False
        self.info = None

class TBRewards(BaseRewards):
    def __init__(self, rewards):
        super().__init__(rewards)


class JOSHReActAgent(BaseAgent):
    def __init__(self, tools, wiki, model: str = "gpt-4-turbo", reason: bool = True, tokenizer=None, debug=False):
        instruction = react_instruction if reason else act_instruction
        self.prompt = wiki + "\n#Available tools\n" + json.dumps(tools) + instruction
        self.model = model
        if create_mode == "llama":
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
        self.temperature = 1.0
        self.debug = debug
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.prompt}]

    

    def act(self, env, index=None, verbose=False, temperature=0.0):
        self.reset()
        obs, info = env.reset(index=index)
        self.messages.append({"role": "user", "content": obs})


        def call_agent(agent, **kwargs):
            """
            Returns the agent and whether it is ready to pass to the customer
            """

            try:
                if self.tokenizer:
                    message, action = get_message_action(
                                    agent.messages, self.model, temperature=self.temperature, tokenizer=self.tokenizer
                                )
                else:
                    message, action = get_message_action(
                                agent.messages, self.model, temperature=self.temperature
                            )
            except Exception as e:
                print(e)
                agent.done = True
                return agent, None

            if not isinstance(action, dict):
                raise TypeError("action must be a dictionary")
            if "name" not in action or not isinstance(action["name"], str):
                raise ValueError("action: 'name' key must be present and must be a string")
            if "arguments" not in action or not isinstance(action["arguments"], dict):
                raise ValueError(
                    "action: 'arguments' key must be present and must be a dictionary"
                )
            
            agent.recent_actions = [action]
            if action["name"] == "respond":
                agent.add_message({"role": "assistant", "content": message})
                return agent, True
            
            obs, reward, done, info = agent.env.step(action)
            obs = "API output: " + obs
            agent.add_message({"role": "assistant", "content": message})
            agent.add_message({"role": "user", "content": obs})

            agent.reward = reward
            agent.done = done
            agent.info = info
            return agent, False
            

        def call_user(user, agent):
            obs, reward, done, info = agent.env.step(agent.recent_actions[0])
            agent.add_message({"role": "user", "content": obs})
            agent.reward = reward
            agent.done = done
            agent.info = info
            return agent, done
        
        def add_error_message(agent):
            agent.add_message({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
            agent.recent_actions = [{"name": 'respond', "arguments": 'Error: Agent ran out of retries.'}]
            return agent



        josh = JOSH(
                rewards=TBRewards(copy.deepcopy(env.task['actions'])), 
                agent_step=call_agent,
                user_step=call_user, 
                add_error_message=add_error_message,
                root_agent = JOSHAgent(copy.deepcopy(self.messages), env),
                user=None,
                debug=self.debug
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
        return max_reward, {'training_examples':josh.training_examples}

    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self):
        return self.messages


react_instruction = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format and have exactly 6 lines (without ```):

```
Thought:
A single line of reasoning to process the context and inform the decision making. Do not include extra lines.
Action:
The name of the action to take. It has to come from "Available tools", or "respond" to respond to the user.
Arguments:
The arguments to the action in json format. If the action is "respond", the argument is the response to the user.
```

You should not use made-up or placeholder arguments.


For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    }
```

Your response can be like this:
```
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
get_current_weather
Arguments:
{"location": "San Francisco, CA", "format": "fahrenheit"}
```

And if the tool returns "70F", your response can be:
```
Thought:
I can answer the user now.
Action:
respond
Arguments:
The current weather of San Francisco is 70F.
```

Try to be helpful and always follow the policy.
"""


act_instruction = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format and have exactly 4 lines (without ```):

```
Action:
The name of the action to take. It has to come from "Available tools", or "respond" to respond to the user.
Arguments:
The arguments to the action in json format. If the action is "respond", the argument is the response to the user.
```

You should not use made-up or placeholder arguments.


For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    }
```

Your response can be like this:
```
Action:
get_current_weather
Arguments:
{"location": "San Francisco, CA", "format": "fahrenheit"}
```

And if the tool returns "70F", your response can be:
```
Action:
respond
Arguments:
The current weather of San Francisco is 70F.
```

Try to be helpful and always follow the policy. Always make sure you generate four lines with the correct format.
"""
