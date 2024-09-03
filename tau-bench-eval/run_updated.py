# Copyright Sierra

import os
import json
import random
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from tau_bench.agents.base import BaseAgent
from tau_bench.envs import get_env

from typing import Any
from math import comb


def build_llama_model():
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    login(token="HF_TOKEN")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
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

    use_peft=True
    #peft_dir = '/home/blattimer/code/multiwoz-api/checkpoints/kto-works-lambdaD=1.7_checkpoint-240/'
    if use_peft:
        peft_dir = '/root/code/multiwoz-api/data/sft/'
        #peft_dir = '/root/multiwoz-api/checkpoints/llama8instructsft-checkpoint-78/'
        from peft import PeftModel, PeftConfig
        model = PeftModel.from_pretrained(model, peft_dir)

    model.eval()
    return model, tokenizer

def run(
    args: argparse.Namespace,
    ckpt_path,
):
    env = get_env(
        args.env,
        user_mode="naive",
        user_model=args.user_model,
        task_split=args.task_split,
    )
    end_index = (
        len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    )
    results = []
    lock = multiprocessing.Lock()
    print(
        f"Running tasks {args.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    if 'llama' in  args.model:
        model, tokenizer = build_llama_model()
    for i in range(args.num_trials):
        idxs = list(range(args.start_index, end_index))
        if args.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> dict:
            isolated_env = get_env(
                args.env,
                user_mode="naive",
                user_model=args.user_model,
                task_split=args.task_split,
            )
            isolated_agent = agent_factory(
                tools_info=env.tools_info,
                wiki=env.wiki,
                args=args,
                model=model,
                tokenizer=tokenizer
            )

            print(f"Running task {idx}")
            try:
                reward, info = isolated_agent.act(
                    isolated_env,
                    idx,
                    verbose=args.verbose,
                    temperature=args.temperature,
                )
                result = {
                    "task_id": idx,
                    "reward": reward,
                    "info": info,
                    "traj": isolated_agent.get_messages(),
                    "trial": i,
                }
            except Exception as e:
                result = {
                    "task_id": idx,
                    "reward": 0,
                    "info": "Error: " + str(e),
                    "traj": isolated_agent.get_messages(),
                    "trial": i,
                }
            print(
                "✅" if result["reward"] == 1 else "❌",
                f"task_id={idx}",
                result["info"],
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




def agent_factory(tools_info, wiki, args: argparse.Namespace, model=None, tokenizer=None) -> BaseAgent:
    # only add think as a tool for function calling
    if not (args.agent_strategy == "function_calling" and args.think):
        tools_info = [
            tool for tool in tools_info if tool["function"]["name"] != "think"
        ]

    if args.agent_strategy == "function_calling":
        if (
            "gpt" in args.model
            or "mistralai/Mi" in args.model
            or "meta-llama/Meta-Llama-3-" in args.model
        ):
            from tau_bench.agents.gpt_function_calling_agent import (
                GPTFunctionCallingAgent,
                initialize_client,
            )

            if "gpt" in args.model:
                initialize_client()
            elif (
                "mistralai/Mi" in args.model or "meta-llama/Meta-Llama-3-" in args.model
            ):
                initialize_client(
                    api_key=os.getenv("ANYSCALE_API_KEY"),
                    base_url="https://api.endpoints.anyscale.com/v1",
                )

            return GPTFunctionCallingAgent(tools_info, wiki, model=args.model)
        elif "claude" in args.model:
            from tau_bench.agents.claude_function_calling_agent import (
                ClaudeFunctionCallingAgent,
            )

            return ClaudeFunctionCallingAgent(tools_info, wiki, model=args.model)
        elif "mistral" in args.model or "mixtral" in args.model:
            from tau_bench.agents.mistral_function_calling_agent import (
                MistralFunctionCallingAgent,
            )

            return MistralFunctionCallingAgent(tools_info, wiki, model=args.model)
        elif "gemini" in args.model:
            from tau_bench.agents.gemini_function_calling_agent import (
                GeminiFunctionCallingAgent,
            )

            return GeminiFunctionCallingAgent(tools_info, wiki, model=args.model)
        else:
            from tau_bench.agents.custom_function_calling_agent import (
                CustomFunctionCallingAgent,
            )

            return CustomFunctionCallingAgent(
                tools_info, wiki, model_name_or_path=args.model, num_gpus=args.num_gpus
            )
    elif args.agent_strategy == "react":
        from tau_bench.agents.chat_react_agent import ChatReActAgent, initialize_create

        if "gpt" in args.model:
            initialize_create(mode="openai")
        elif "llama" in args.model:
            initialize_create(mode="llama")
            return ChatReActAgent(tools_info, wiki, model=model, reason=args.think, tokenizer=tokenizer)
        elif "claude" in args.model:
            initialize_create(mode="anthropic")
        elif "gemini" in args.model:
            initialize_create(mode="google")
        else:  # anyscale
            initialize_create(
                mode="openai",
                api_key=os.getenv("ANYSCALE_API_KEY"),
                base_url="https://api.endpoints.anyscale.com/v1",
            )
        return ChatReActAgent(tools_info, wiki, model=args.model, reason=args.think)
    else:
        raise ValueError(f"Unknown agent strategy: {args.agent_strategy}")

def display_metrics(results: dict[str, Any]) -> None:
    num_trials = len(set([r["trial"] for r in results]))
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for r in results:
        if r["task_id"] not in c_per_task_id:
            c_per_task_id[r["task_id"]] = r["reward"]
        else:
            c_per_task_id[r["task_id"]] += r["reward"]
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=[
            # openai api models
            "gpt-4-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-4o",
            # anthropic api models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            # google api models
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro",
            # mistral api models,
            "open-mixtral-8x22b",
            "mistral-large-latest",
            # anyscale api models
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "llama"
        ],
    )
    parser.add_argument(
        "--user_model",
        type=str,
        default="gpt-4",
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        default="function_calling",
        choices=["function_calling", "react"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task_split", type=str, default="test", choices=["train", "test", "dev"]
    )
    parser.add_argument(
        "--think", type=int, default=0, help="Add think for function calling"
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    time_str = datetime.now().strftime("%m%d%H%M%S")
    file_str = f"{args.log_dir}/{args.agent_strategy}{args.think}-{args.model.split('/')[-1]}-{args.temperature}_range_{args.start_index}-{args.end_index}_user{args.user_model}_{time_str}.json"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    results = run(
        args=args,
        ckpt_path=file_str,
    )

    display_metrics(results)


    with open(file_str, "w") as f:
        json.dump(results, f, indent=2)
        rewards = [r["reward"] for r in results]
        print(f"🏆 Average reward: {sum(rewards) / len(rewards)}")
        print(f"\n📄 Results saved to {file_str}\n")


if __name__ == "__main__":
    main()
