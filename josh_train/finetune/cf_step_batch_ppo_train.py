from datetime import datetime
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch
import numpy as np
from typing import Dict, List, Tuple
import wandb
from dataclasses import dataclass
from torch.utils.data import Dataset
import argparse
from josh_train.utils import *
from openai import OpenAI
import josh_train.config as config
from josh_train.users.goal_user_simulator import GoalUserSimulator
from josh_train.main import ToolWOZEnvironment, build_user, build_convo_env, build_agent, ToolWOZRewards
from torch.optim import AdamW
from josh_train.conversation_types.conversation_state_pref_tree import Conversation
import gc
import time
import torch.nn.utils.rnn as rnn_utils
@dataclass
class ConversationTurn:
    query: str
    response: str
    reward: float


import math
from typing import Optional
import yaml
import re
import copy
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from josh_train.utils import make_transcript, request_openai, parse_api_call, handle_api_calls
import os
import josh_train.config as config
from josh_train.josh import BaseJOSHAgent
import bitsandbytes as bnb

class LocalReACTAgentSimulator(BaseJOSHAgent):
    def __init__(self, api_examples, api_defs, tokenizer, model_name:Optional[str]=None, temperature=0.0, debug = False):
        super().__init__()
        cwd = os.getcwd()
        with open(f'{cwd}/prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}
        with open(f'{cwd}/data/tools.json', 'r') as file:
            tools_list = json.load(file)
        self.MONO_PROMPT = prompts['maybe_fine_prompt']#.replace('{example_filled}', json.dumps(tools_list, indent=2))
        self.pattern = "(PLAN|APICALL|SPEAK)(.*?)(?=PLAN|APICALL|SPEAK|$)"
        self.model_name=model_name
        self.debug = debug
        self.temperature = temperature
        self.SHORT_PROMPT = prompts['maybe_fine_prompt']#.replace('{example_filled}', json.dumps(tools_list, indent=2))
        self.system_short_enc = tokenizer(
            self.SHORT_PROMPT, return_tensors="pt", add_special_tokens=False
        ).input_ids.to('cuda')

    def parse_agent_message(self, output):
        commands  = re.findall(self.pattern , output , re.DOTALL)
        return commands
    
    
    def request(self, messages, model=None, tokenizer=None) -> str:
        # print('request')
        model.model.gradient_checkpointing_disable()
        model.model.eval()
        t0 = time.time()
        with torch.no_grad():
            print(f'message: {"sys" if len(messages)<=1 else messages[1:]}')
            print(f'msg len: {len(messages)}')
            # dynamic_messages = messages[1:]  # Skip system prompt
            for msgs_idx in range(1, len(messages)):
                if msgs_idx > 1:
                    print(f'truncate {msgs_idx-1} message')
                dynamic_messages = tokenizer.apply_chat_template(messages[msgs_idx:], return_tensors="pt")
                if dynamic_messages.shape[-1]+self.system_short_enc.shape[-1]<1175:
                    break
            dynamic_enc = dynamic_messages.to('cuda')
            # output_short_encoding = torch.cat([self.system_short_enc, dynamic_enc], dim=-1).squeeze()
            encoding = torch.cat([self.system_short_enc, dynamic_enc], dim=-1).squeeze()
            print(encoding.shape)

            prompt_len = encoding.shape[-1]

            with torch.inference_mode():
                generated_ids = model.generate(encoding, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)

            return_ids = generated_ids[:, prompt_len:]
            output_text = tokenizer.decode(return_ids[0], skip_special_tokens=True)
            
        t1 = time.time()
        print(f'inf time: {t1-t0:.3f} seconds')
        return output_text, encoding, return_ids
    
    def handle_api(self, command, conversation_state):
        try:
            api_values = parse_api_call(command)
        except:
            return 'FAILURE INCORRECTLY FORMATTED APICALL', None
        if api_values['api_name'] not in self.apis_to_examples:
            return 'FAILURE INCORRECTLY FORMATTED APICALL', None
        returns = handle_api_calls(api_values['api_name'], api_values['api_args'], conversation_state=conversation_state)
        if type(returns)==list:
            called_api = {'name':api_values['api_name'], 'parameters': api_values['api_args'], 'returned': returns[0] if len(returns)>0 else returns}
        else:
            called_api = {'name':api_values['api_name'], 'parameters': api_values['api_args'], 'returned': returns}
        return returns, called_api
    
    def step(self, model, **kwargs):
        conversation_state = kwargs['env']
        tokenizer = kwargs['tokenizer']
        training_outputs = []
        self.recent_actions = []
        output_mask=[]
        count=0
        while count < 3:
            agent_messages = [{'role':'system', 'content':self.MONO_PROMPT}]+self.messages_internal
            turn, input_ids, response_ids = self.request(agent_messages, model, tokenizer)
            output_mask_val = True
            #make sure they're not zero dim
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if response_ids.dim() == 0:
                response_ids = response_ids.unsqueeze(0)
            #add to train outputs
            training_outputs.append((input_ids, response_ids))
            if self.debug:
                print(turn)
            parsed = self.parse_agent_message(turn.replace('<COMMAND_END>', '').strip().replace('\n','').replace('\\',''))
            if len(parsed)==0:
                self.messages_internal.append({'role':'assistant', 'content':'ERROR: NO COMMAND FOUND'})
                
            thought_string = ''
            for command_type, command in parsed:
                command_type = command_type.strip()
                command=command.strip()
                if command_type=='PLAN':
                    thought_string = 'PLAN '+command+' <COMMAND_END> '
                elif command_type == 'SPEAK':
                    self.messages_internal.append({'role':'assistant', 'content':thought_string+'SPEAK '+command+' <COMMAND_END>'})
                    self.messages.append({'role':'assistant', 'content':command})
                    output_mask.append(True)
                    return training_outputs, output_mask
                elif command_type == 'APICALL':
                    command = command.strip().replace('\n','')
                    output, called_api = self.handle_api(command, conversation_state)
                    self.recent_actions.append(called_api)
                    if self.debug:
                        print(output)
                    # Add the api call
                    self.messages_internal.append({'role':'assistant', 'content':thought_string+'APICALL '+command+' <COMMAND_END>'})
                    # Add the return
                    self.messages_internal.append({'role':'user', 'content':'APIRETURN ' + json.dumps(output)})
                    # if there was an error, make sure reward is masked to zero
                    if type(json.dumps(output)) is str:
                        output_mask_val = False
                else:
                    self.messages_internal.append({'role':'assistant', 'content':'ERROR: INVALID COMMAND TYPE'})
                    output_mask_val = False
            output_mask.append(output_mask_val)
            count+=1
        self.messages.append({'role':'assistant', 'content':'Error: Agent ran out of retries.'})
        return training_outputs, output_mask

class ConversationDataset(Dataset):
    def __init__(self, conversations: List[ConversationTurn]):
        self.conversations = conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]


class PPOToolWOZTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_models()
        self.setup_ppo_config()
        self.env = ToolWOZEnvironment(args)
        self.initialize_trainer()
        self.smooth_gamma = 0.95  # Decay factor for reward propagation

    # def _smooth_rewards(self, rewards: List[float], out_mask: List[bool]) -> List[float]:
    #     """Apply exponential decay to propagate rewards backward"""
    #     smoothed = torch.zeros(len(rewards))
    #     carryover = 0.0
    #     for i in reversed(range(len(rewards))):
    #             carryover = carryover * self.smooth_gamma + rewards[i]
    #             # only allow carryover if there wasn't a failed api call
    #             if out_mask[i]:
    #                 smoothed[i] = carryover
    #     if len(rewards) > 0 and smoothed.max() > 0:
    #             smoothed = torch.tensor([min(1.0, x) for x in smoothed])
    #     return smoothed.tolist()

    def _smooth_rewards(self, rewards: List[float], out_mask: List[bool]) -> List[float]:
        smoothed = torch.zeros(len(rewards))
        carryover = 0.0
        for i in reversed(range(len(rewards))):
            carryover = carryover * self.smooth_gamma + rewards[i]
            # Modified: Allow partial carryover even on failures
            if out_mask[i]:
                smoothed[i] = carryover
            else:
                smoothed[i] = carryover * 0.3  # Penalize less harshly
        # Remove min(1.0) cap to allow meaningful scaling
        if len(rewards) > 0:
            smoothed /= smoothed.max() + 1e-6  # Normalize instead
        smoothed_list = smoothed.tolist()
        smoothed_scaled = [r * 0.1 for r in smoothed_list]
        return smoothed_scaled
        
    def setup_models(self):
        # Setup quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model with value head
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        # 3. Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        # peft_model = get_peft_model(base_model, lora_config)
        
        print('valhead')
        # 4. Add value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                                                       peft_config=lora_config,
                                                                       attn_implementation="flash_attention_2",
                                                                       use_flash_attention_2=True,  # Critical for long sequences
                                                                       quantization_config=self.bnb_config,
                                                                       device_map="cuda",
                                                                       torch_dtype=torch.bfloat16, 
                                                                    use_cache=True,)#, quantization_config=self.bnb_config, torch_dtype=torch.bfloat16, device_map="cuda",)
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.requires_grad_(False)
        # 5. Verify value head parameters
        for name, param in self.model.named_parameters():
            if "v_head" in name or "lora" in name:
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params}, Trainable: {trainable_params}")
        # print([n for n,p in self.model.named_parameters() if p.requires_grad])


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Better for generation
        
        # Enable gradient checkpointing
        # self.model.gradient_checkpointing_enable()
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            quantization_config=self.bnb_config,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, 
            use_cache=True,
            use_flash_attention_2=True,
        )
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        


    def setup_ppo_config(self):
        self.ppo_config = PPOConfig(
            learning_rate=1e-4,  # Increased from 5e-5
            batch_size=64,       # Larger batch size for stability
            mini_batch_size=2,   # Critical! Avoids noisy updates
            gradient_accumulation_steps=4,
            ppo_epochs=4,             # Fewer epochs to avoid overfitting
            cliprange=0.1,            # Tighter clipping for stability
            cliprange_value=0.1,      # Clip value updates aggressively
            vf_coef=0.5,              # Prioritize value learning
            init_kl_coef=0.1,         # Penalize large KL divergences
            target_kl=0.05,           # Enforce early stopping
            gamma=0.99,
            lam=0.95,
            gradient_checkpointing=True,
            optimize_cuda_cache=True,
            remove_unused_columns=True,
            is_peft_model=True,
            early_stopping=True,
            max_grad_norm=0.3,
            log_with = "wandb",

        )

    def initialize_trainer(self):

        params = [p for p in self.model.parameters() if p.requires_grad]
        # print(params)
        # Create custom optimizer
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=self.ppo_config.learning_rate,
            weight_decay=0.00,
            eps=1e-6,
            betas=(0.9, 0.999),
        )
        from transformers import get_cosine_schedule_with_warmup
    
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=10000,
        )
        
        # Initialize trainer with custom optimizer
        self.trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            optimizer=optimizer,  # Pass custom optimizer
            dataset=None,
            lr_scheduler=scheduler
        )

    def train_step(self, observations, actions, rewards_list, step_num):
        self.trainer.model.gradient_checkpointing_enable()
        self.trainer.model.train()
        rewards_tensor = [torch.tensor(x, device="cuda") for x in rewards_list]
        print(f"Max input len {observations[-1].shape}")
        print(f"Memory before step: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"(Before) Max memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
        # Run PPO Step
        from torch.nn.utils.rnn import pad_sequence
        ppo_stats = self.trainer.step(observations, actions, rewards_tensor)
        print(f"(After) Max memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
        print(ppo_stats.keys())
        return {
            # Core PPO metrics
            "ppo/loss/total": ppo_stats["ppo/loss/total"],
            "ppo/loss/policy": ppo_stats["ppo/loss/policy"],
            "ppo/loss/value": ppo_stats["ppo/loss/value"],
            "ppo/policy/entropy": ppo_stats["ppo/policy/entropy"],
            "ppo/policy/approxkl": ppo_stats["ppo/policy/approxkl"],
            "ppo/policy/clipfrac": ppo_stats["ppo/policy/clipfrac"],
            
            # Value function metrics
            "ppo/val/error": ppo_stats["ppo/val/error"],
            "ppo/val/var_explained": ppo_stats["ppo/val/var_explained"],
            
            # Reward statistics
            "ppo/mean_scores": ppo_stats["ppo/mean_scores"],
            "ppo/returns/mean": ppo_stats["ppo/returns/mean"],
            "ppo/returns/var": ppo_stats["ppo/returns/var"],
            
            # KL divergence metrics
            "objective/kl": ppo_stats["objective/kl"],
            "objective/kl_coef": ppo_stats["objective/kl_coef"],
            
            # Token statistics
            "tokens/queries_len_mean": ppo_stats["tokens/queries_len_mean"],
            "tokens/responses_len_mean": ppo_stats["tokens/responses_len_mean"],
            
            # Performance metrics
            "time/ppo/total": ppo_stats["time/ppo/total"],

            # "env/reward_mean": ppo_stats["env/reward_mean"],
            # "env/reward_std": ppo_stats["env/reward_std"],
            # "env/reward_dist": ppo_stats["env/reward_dist"],
            # "ppo/mean_non_score_reward": ppo_stats["ppo/mean_non_score_reward"],
            # "ppo/mean_non_score_reward": ppo_stats["ppo/mean_non_score_reward"],
            # "ppo/policy/ratio": ppo_stats["ppo/policy/ratio"],
        }
    
    def build_env(self, conversation_id: str):
        convo_env = Conversation(conversation_id, self.env.apis, self.env.delex)
        user = GoalUserSimulator(goals=convo_env.goals, modelname=self.args.user_model, debug=self.args.debug)
        agent = LocalReACTAgentSimulator(self.env.api_examples, self.env.valid_api_defs, model_name=None,
                                        debug=self.args.debug, temperature=self.args.temperature,
                                        tokenizer=self.tokenizer)

        rewards = ToolWOZRewards(convo_env)

        return {'convo_env':convo_env,
                'user':user,
                'agent':agent,
                'rewards':rewards,
                'max_reward': len(rewards),
                'observations':[],
                'actions':[],
                'rewards_list':[],
                'out_mask':[],
                'convo_over':False,
                'total_rewards':0.0
                }


    def step_conversation(self, convo_env):
        with torch.no_grad():
            convo_env['agent'], convo_over = convo_env['user'].step(convo_env['agent'])
            if convo_over:
                convo_env['convo_over']=True
                return convo_env
            training_outputs, output_mask = convo_env['agent'].step(self.trainer, tokenizer=self.tokenizer, env=convo_env['convo_env'])

            convo_env['out_mask'] += output_mask
            got_reward, rw_to_delete = convo_env['rewards'].is_reward(convo_env['agent'].recent_actions)
            reward = 1.0 if got_reward else 0.0
            convo_env['total_rewards'] += 1.0 if got_reward else 0.0
            if got_reward:
                convo_env['rewards'].delete_reward(rw_to_delete)

            for idx, x in enumerate(training_outputs):
                input_tensor = x[0].squeeze()
                response_tensor = x[1].squeeze()

                convo_env['observations'].append(input_tensor)
                convo_env['actions'].append(response_tensor)
                convo_env['rewards_list'].append(reward) 

            return convo_env
        
    def make_n_environments(self, train_ids, n):
        envs = []
        for i in range(n):
            id = train_ids.pop()
            envs.append(self.build_env(id))
        return envs, train_ids
    
    def get_current_len(self, envs):
        return sum([len(x['observations']) for x in envs])
    
    def smooth_env_rewards(self, envs):
        for i in range(len(envs)):
            envs[i]['rewards_list'] = self._smooth_rewards(envs[i]['rewards_list'], envs[i]['out_mask'])
        return envs
    
    def make_batch(self, envs, shuffle=True):
        batch_obs = []
        batch_actions = []
        batch_rewards = []

        for env in envs:
            batch_obs += env["observations"]
            batch_actions += env["actions"]
            batch_rewards += env["rewards_list"]

        if shuffle:
            indices = list(range(len(batch_obs)))
            random.shuffle(indices)
            batch_obs = [batch_obs[i] for i in indices]
            batch_actions = [batch_actions[i] for i in indices]
            batch_rewards = [batch_rewards[i] for i in indices]

        return batch_obs, batch_actions, batch_rewards

    
    
    def step_n_environments(self, envs, train_ids):
        for i in range(len(envs)):
            if not envs[i]["convo_over"]:
                envs[i] = self.step_conversation(envs[i])
                if envs[i]["convo_over"]:
                    id = train_ids.pop()
                    envs.append(self.build_env(id))
        return envs, train_ids

    def train(self, num_epochs: int = 10):
        wandb.init(project="toolwoz-ppo")
        with open('/root/josh-llm-simulation-training/training_fnames.json', 'r') as file:
            training_names = json.load(file)
        time_str = datetime.now().strftime("%m%d%H%M%S")
        train_ids = sorted(training_names)
        np.random.seed(42)
        np.random.shuffle(train_ids)

        if not os.path.exists(f"run_{time_str}"):
            os.makedirs(f"run_{time_str}")
        stopping_number = self.trainer.config.batch_size
        train_num = 0
        while len(train_ids)>=5:
            envs, train_ids = self.make_n_environments(train_ids=train_ids, n=8)

            current_len=0
            while current_len<stopping_number:
                envs, train_ids = self.step_n_environments(envs, train_ids=train_ids)
                current_len = self.get_current_len(envs)
                # print(self.get_current_len(envs))
                # print(f'j:{j}')
                
            envs = self.smooth_env_rewards(envs)
            # print(len(envs))
            
            batch_obs, batch_actions, batch_reward = self.make_batch(envs, shuffle=True)
            # print(batch_obs)
            log_msg = self.train_step(batch_obs[:stopping_number], batch_actions[:stopping_number], batch_reward[:stopping_number], step_num=train_num)
            wandb.log(log_msg, step=train_num)

            # Save checkpoint
            if (train_num) % 5 == 0:
                output_dir = f"run_{time_str}/ppo_checkpoint_epoch_{train_num}_{time_str}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.trainer.model.save_pretrained(
                    output_dir, state_dict=self.trainer.model.state_dict(), safe_serialization=True
                )
                # self.trainer.model.save_pretrained(f"run_{time_str}/ppo_checkpoint_epoch_{train_num}_{time_str}")
            train_num  +=1

                
        wandb.finish()

def main():
    creds = get_openai_creds()
    api_key = creds['openai_key']
    api_org = creds['openai_org']
    config.client = OpenAI(api_key=api_key, organization=api_org)

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
    
    trainer = PPOToolWOZTrainer(args)
    trainer.train()

    

if __name__ == "__main__":
    main()

