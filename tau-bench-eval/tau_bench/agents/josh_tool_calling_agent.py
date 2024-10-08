# Copyright Sierra

import copy
import json
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
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

class JOSHToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        debug: bool = False,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.debug = debug

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        
        def call_agent(agent, **kwargs):
            res = completion(
                messages=agent.messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            agent.total_cost += res._hidden_params["response_cost"] if res._hidden_params["response_cost"] else 0.0
            action = message_to_action(next_message)
            agent.recent_actions = [action]

            if action.name == RESPOND_ACTION_NAME:
                agent.add_message(next_message)
                return agent, True

            env_response = agent.env.step(action)
            agent.reward = env_response.reward
            agent.info = {**agent.info, **env_response.info.model_dump()}
            agent.done = env_response.done


            next_message["tool_calls"] = next_message["tool_calls"][:1]


            agent.add_message({
                    'role':'assistant',
                    'tool_calls' : [ {
                        'id'       : next_message["tool_calls"][0]["id"],
                        'type'     : 'function',
                        'function' : {
                            'name'      : next_message["tool_calls"][0]["function"]["name"],
                            'arguments' : next_message["tool_calls"][0]["function"]["arguments"]
                        }
                    } ]
                }
            )
            agent.add_message(
                {
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "content": env_response.observation,
                }
            )

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
            agent.recent_actions = [{"name": 'respond', "arguments": 'Error: Agent ran out of retries.'}]
            return agent

        ### Set up first message
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        info = env_reset_res.info.model_dump()

        root_agent = JOSHAgent(copy.deepcopy(messages), env, env_reset_res)

        ### Set up JOSH

        josh = JOSH(
                rewards=TBRewards(copy.deepcopy(env.task.actions)), 
                agent_step=call_agent,
                user_step=call_user, 
                add_error_message=add_error_message,
                root_agent = root_agent,
                user=None,
                debug=self.debug
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
    

def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
