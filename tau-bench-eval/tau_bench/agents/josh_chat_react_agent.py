# Copyright Sierra

import copy
import json
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple
from josh_train.josh import JOSH, BaseJOSHAgent, BaseRewards

class JOSHAgent(BaseJOSHAgent):
    def __init__(self, messages, env, env_reset_res):
        super().__init__(messages=messages)
        self.env = env
        self.reward = 0
        self.done = False
        self.info = env_reset_res.info.model_dump()
        self.total_cost = 0.0

class TBRewards(BaseRewards):
    def __init__(self, rewards):
        super().__init__(rewards)


class JOSHChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        debug: bool = False,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.debug = debug

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        return message.model_dump(), action, res._hidden_params["response_cost"]


    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        
        def call_agent(agent, **kwargs):
            message, action, cost = self.generate_next_step(agent.messages)

            agent.total_cost += cost
            agent.recent_actions = [action]

            if action.name == RESPOND_ACTION_NAME:
                agent.add_message(message)
                return agent, True

            env_response = agent.env.step(action)
            agent.reward = env_response.reward
            agent.info = {**agent.info, **env_response.info.model_dump()}
            agent.done = env_response.done

            agent.add_message(message)
            agent.add_message({"role": "user", "content": "API output: " + env_response.observation})

            return agent, False


        def call_user(user, agent):
            env_response = agent.env.step(agent.recent_actions[0])
            agent.add_message({"role": "user", "content": env_response.observation})
            agent.reward = env_response.reward
            agent.done = env_response.done
            agent.info = {**agent.info, **env_response.info.model_dump()}
            return agent, env_response.done

        def add_error_message(agent):
            agent.add_message({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
            agent.recent_actions = [Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME:'Error: Agent ran out of retries.'})]
            return agent



        ### Set up first message
        env_reset_res = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": env_reset_res.observation},
        ]
        total_cost = 0.0
        info = {}
        root_agent = JOSHAgent(copy.deepcopy(messages), env, env_reset_res)

        ### Set up JOSH

        josh = JOSH(
                rewards=TBRewards(copy.deepcopy(env.task.actions)), 
                agent_step=call_agent,
                user_step=call_user, 
                add_error_message=add_error_message,
                root_agent = root_agent,
                user=None,
                debug=self.debug,
                max_turn_tries=10,
                beam_size=8
            )
        
        ### Start running
        for _ in range(max_num_steps):
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
            reward_info = josh.golden_agent.env.calculate_reward()
            max_reward = reward_info.reward
            info = josh.golden_agent.info

        training_examples = []
        for ex in josh.training_examples:
            example = ({'messages':ex[0]}, ex[1], ex[2])
            example[0]['tools'] = self.tools_info
            training_examples.append(example)
        info['training_examples'] = training_examples
        return SolveResult(
            reward=max_reward,
            info=info,
            messages=[{}],
            total_cost=total_cost,
        )


REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
