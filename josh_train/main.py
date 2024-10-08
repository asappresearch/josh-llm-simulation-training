import argparse
import copy
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import math
import random
from josh_train.josh import BaseJOSHAgent, JOSH, BaseRewards
from josh_train.utils import *
from openai import OpenAI
import josh_train.config as config

class ToolWOZEnvironment:
    def __init__(self, args):
        # Build the environment
        dbs = create_dbs()
        if not os.path.isfile('data/ground_truth_apis.json'):
            self.apis = create_apis(dbs)
            with open('data/ground_truth_apis.json', 'w') as file:
                json.dump(self.apis, file, indent=2)
        else:
            with open('data/ground_truth_apis.json', 'r') as file:
                self.apis = json.load(file)

        if not os.path.isfile('data/api_examples.json'):
            self.api_examples = create_api_examples(self.apis)
            with open('data/api_examples.json', 'w') as file:
                json.dump(self.api_examples, file, indent=2)
        else:
            with open('data/api_examples.json', 'r') as file:
                self.api_examples = json.load(file)


        with open('data/valid_api_defs.json', 'r') as file:
            self.valid_api_defs = json.load(file)

        with open('data/delex.json') as outfile:
            self.delex = json.load(outfile)

        with open('data/data.json') as outfile:
            self.real_convos = json.load(outfile)

        with open('data/testListFile.json', 'r') as file:
            test_ids_tmp = file.readlines()
        test_ids_tmp = [x.strip() for x in test_ids_tmp]
        self.test_ids_full = sorted(list(set(test_ids_tmp).intersection(self.apis.keys())))
        self.test_ids = sorted(list(set(test_ids_tmp).intersection(self.apis.keys())))[:450]
        with open('data/valListFile.json', 'r') as file:
            val_ids_tmp = file.readlines()
        val_ids_tmp = [x.strip() for x in val_ids_tmp]
        self.val_ids = sorted(list(set(val_ids_tmp).intersection(self.apis.keys())))

        self.train_ids = sorted(list(set([k for k in self.apis.keys()]) - set(self.test_ids_full) - set(self.val_ids)))

        if args.task_split =='train':
            self.set_to_run = self.train_ids
        elif args.task_split =='val':
            self.set_to_run = self.val_ids
        elif args.task_split =='full_test':
            self.set_to_run = self.test_ids_full
        else:
            self.set_to_run = self.test_ids

        self.model =None
        self.tokenizer=None
        if 'llama' in args.model.lower():
            self.model, self.tokenizer = self.build_hf_model(args)
    
    def build_hf_model(self, args):
        from huggingface_hub import login
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        hf_creds = get_hf_creds()
        login(token=hf_creds["hf_token"])

        model_name = args.model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)#, padding_side="left")

        if args.peft_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.peft_dir)
            #Don't do merge and unload, it ruins things!!
            #model = model.merge_and_unload()

        model.eval()

        return model, tokenizer

def build_convo_env(args, convo_key, toolwoz_env):
    if args.josh:
        from josh_train.conversation_types.conversation_state_pref_tree import Conversation
    else:
        from josh_train.conversation_types.conversation_state import Conversation

    convo_state = Conversation(
                    convo_key, 
                    toolwoz_env.apis, 
                    toolwoz_env.delex
                )
    return convo_state

def build_user(args, toolwoz_env, conversation_env):
    if args.user_mode =='guide':
        from josh_train.users.guide_user_simulator import GuideUserSimulator
        user = GuideUserSimulator(goals=conversation_env.goals, 
                             convo=toolwoz_env.real_convos[conversation_env.convo_key],
                             modelname=args.user_model,
                             debug=args.debug
                            )
    else:
        from josh_train.users.goal_user_simulator import GoalUserSimulator
        user = GoalUserSimulator(goals=conversation_env.goals, 
                             modelname=args.user_model,
                             debug=args.debug
                            )
    return user

def build_agent(args, toolwoz_env):
    if args.agent_strategy == 'react':
        from josh_train.agents.react_agent import ReACTAgentSimulator
        agent = ReACTAgentSimulator(
                        toolwoz_env.api_examples, 
                        toolwoz_env.valid_api_defs,
                        model_name=args.model,
                        debug=args.debug,
                        temperature= args.temperature,
                        )
    elif args.agent_strategy == 'function_calling':
        from josh_train.agents.fc_agent import FCAgentSimulator
        agent = FCAgentSimulator(
                        toolwoz_env.api_examples, 
                        toolwoz_env.valid_api_defs,
                        model_name=args.model, 
                        debug=args.debug,
                        temperature = args.temperature
                        )
    else:
        raise ValueError(f'Agent strategy {args.agent_strategy} not supported')
    return agent

def _run_conversation_normal(args, agent, user, convo_env, toolwoz_env):
    while len(agent.messages)<args.max_convo_turns:
        agent, convo_over = user.step(agent)
        if convo_over:
            break
        agent.step(model=toolwoz_env.model, tokenizer=toolwoz_env.tokenizer, env=convo_env)
    reward, failed_api = convo_env.evaluate_apis()

    return reward, {'messages':agent.messages, 'failed_apis':failed_api, 'internal_messages':agent.messages_internal}

class ToolWOZRewards(BaseRewards):
    def __init__(self, env):
        self.env = env
        self.rewards = []
        for domain in env.apis_for_eval.keys():
            for api in env.apis_for_eval[domain]['success'].keys():
                self.rewards.append((domain, api))
        super().__init__(self.rewards)

    def is_reward(self, agent_actions):
        filtered_actions = [a for a in agent_actions if a]
        correct_calls, _, _, apis_to_delete = self.env.evaluate_apis(filtered_actions)
        got_reward = correct_calls>0

        return got_reward, apis_to_delete
    
    def delete_reward(self, rewards_to_delete):
        for reward_to_delete in rewards_to_delete:
            self.rewards.remove(reward_to_delete)
            del self.env.apis_for_eval[reward_to_delete[0]]['success'][reward_to_delete[1]]


def _run_conversation_josh(args, agent, user, convo_env, toolwoz_env):
    agent, _ = user.step(agent)
    def add_error_message(agent):
        agent.messages.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return agent
    
    def step_agent(agent:BaseJOSHAgent, **kwargs):
        agent.step(**kwargs)
        return agent, True
    
    def step_user(user, agent:BaseJOSHAgent):
        return user.step(agent)
    
    josh = JOSH(
                rewards=ToolWOZRewards(convo_env),
                agent_step=step_agent,
                user_step=step_user,
                add_error_message=add_error_message,
                root_agent = agent,
                user = user,
                agent_model = toolwoz_env.model,
                agent_tokenizer=toolwoz_env.tokenizer,
                agent_env=convo_env,
                beam_size=args.beam_size,
                max_turn_tries=args.josh_agent_tries,
                debug=args.josh_debug,
            )
    for _ in range(15):
        try:
            max_reward, all_done = josh.step()
        except Exception as e:
            print(e)
            break
        if all_done:
            break

    training_examples = []
    for ex in josh.training_examples:
        example = ({'messages':ex[0]}, ex[1], ex[2])
        if args.agent_strategy == 'function_callling':
            example[0]['tools'] = agent.tool_list
        training_examples.append(example)

    return max_reward, {'training_examples':training_examples}

def run_conversation(args, agent, user, convo_env, toolwoz_env):
    if not args.josh:
        return _run_conversation_normal(args, agent, user, convo_env, toolwoz_env)
    else:
        return _run_conversation_josh(args, agent, user, convo_env, toolwoz_env)

def driver(
    args: argparse.Namespace,
    ckpt_path,
):
    # Get openai creds to play the user
    creds = get_openai_creds()
    api_key = creds['openai_key']
    api_org = creds['openai_org']
    config.client = OpenAI(api_key=api_key, organization=api_org)

    toolwoz_env = ToolWOZEnvironment(args)

    end_index = (
        len(toolwoz_env.set_to_run) if args.end_index == -1 else min(args.end_index, len(toolwoz_env.set_to_run))
    )
    results = []
    lock = multiprocessing.Lock()
    print(
        f"🏃🏃🏃 Simulating {args.task_split} convos from {args.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    total_idx = list(range(len(toolwoz_env.set_to_run)))
    if args.shuffle:
        random.shuffle(total_idx)

    idxs = total_idx[args.start_index: end_index]

    def _run(idx: int) -> dict:
        convo_env = build_convo_env(args, toolwoz_env.set_to_run[idx], toolwoz_env)
        user = build_user(args, toolwoz_env, convo_env)
        agent = build_agent(args, toolwoz_env)

        print(f"Running simulation {idx} ({toolwoz_env.set_to_run[idx]})")
        try:
            reward, extra_data = run_conversation(args, agent, user, convo_env, toolwoz_env)
            result =  {
                'id':toolwoz_env.set_to_run[idx],
                'reward':reward,
                **extra_data,
            }
        except Exception as e:
            print("Error: " + str(e))
            reward=0.0
            messages = []
            failed_api=[]
            training_examples=[]
            result = {
                'id':toolwoz_env.set_to_run[idx],
                'reward':reward,
                'failed_apis':failed_api,
                'messages':messages,
                'training_examples':training_examples,
                'error':"Error: " + str(e)
            }
        convo_env.close_convos()
        convo_env = None
        del convo_env

        if math.isclose(reward, 1.0, rel_tol=1e-6):
            output_emoji = "🟢"
        elif math.isclose(reward, 0.0, rel_tol=1e-6):
            output_emoji = "🔴"
        else:
            output_emoji = "🟡"
        print(
            output_emoji,
            f"task_id={idx} ({toolwoz_env.set_to_run[idx]})",
            f"reward={round(reward,2)}"
        )
        print("-----")
        with lock:
            data = []
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    data = json.load(f)
            with open(ckpt_path, "w") as f:
                json.dump(data + [result], f, indent=2)
        return result

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        res = list(executor.map(_run, idxs))
        results.extend(res)

    return results


def final_metric(results):
    rewards = [r["reward"] for r in results]
    avg_reward = round(sum(rewards) / len(rewards), 4)
    success_rate = round(len([r for r in rewards if math.isclose(r, 1.0, rel_tol=1e-6)]) / len(rewards), 4)
    print(f"📊 Average Reward: {avg_reward}")
    print(f"🏆 100% Success Rate: {success_rate}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--peft_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--user_mode",
        type=str,
        default="goal",
        choices=["goal", "guide"],
    )
    parser.add_argument(
        "--user_model",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        default="react",
        choices=["function_calling", "react"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task_split", type=str, default="test", choices=["train", "test", "val", "full_test"]
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--max_convo_turns", type=int, default=15, help="Number of user/agent turns that can run in one convo")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--josh_debug", action="store_true", default=False)
    parser.add_argument("--josh", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="records")
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--josh_agent_tries", type=int, default=20)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    if args.josh and math.isclose(args.temperature, 0.0, rel_tol=1e-6):
        raise ValueError('JOSH should be ran with nonzero temperature')

    time_str = datetime.now().strftime("%m%d%H%M%S")
    file_str = f"{args.log_dir}/{args.agent_strategy}-{args.model.split('/')[-1]}-{args.temperature}_range_{args.start_index}-{args.end_index}_user{args.user_model}_{time_str}.json"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    results = driver(
        args=args,
        ckpt_path=file_str,
    )

    final_metric(results)


    with open(file_str, "w") as f:
        json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to {file_str}\n")


if __name__ == "__main__":
    main()
