import json
import os
from datasets import Dataset, DatasetDict 
import pandas as pd
from trl import KTOConfig, KTOTrainer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import yaml
import os

path = os.getcwd()

with open(f'{path}/prompts/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)
with open(f'{path}/data/tools.json', 'r') as file:
    tools_list = json.load(file)
MONO_PROMPT = prompts['mistral_mono_prompt'].replace('{example_filled}', json.dumps(tools_list, indent=2))


from huggingface_hub import login
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

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# from peft import PeftModel, PeftConfig
# model = PeftModel.from_pretrained(model, '/home/blattimer/code/multiwoz-api/data/llama_8b_ft_0501/checkpoint-240')
# model = model.merge_and_unload()
model.train()

folder = f'{path}/records/training_data/'#/multiwoz-api/records/llama_sims/'#folder name
file_list = [folder+x for x in os.listdir(folder) if '.json' in x]
file_list = sorted(file_list, key=lambda x: x.split('.json')[0].split('_')[-1])
files = []
for fname in file_list:
    with open(fname, 'r') as file:
        files.append(json.load(file)['training_paths'])


import re
def parse_agent_message(output):
    pattern = "(THOUGHTS|APICALL|SPEAK)(.*?)(?=THOUGHTS|APICALL|SPEAK|$)"
    commands  = re.findall(pattern , output , re.DOTALL)
    return commands

def clean_good_utterance(utterance, resulting_utt=None):
    if 'APIRETURN' in utterance:
        return False
    if 'ERROR:' in utterance:
        return False
    if 'Error:' in utterance:
        return False
    if 'APICALL' in utterance:
        try:
            json.loads(parse_agent_message(utterance)[1][1].replace('<COMMAND_END>','').strip())
            if resulting_utt:
                assert 'FAILURE' not in resulting_utt['content']
        except:
            return False
    return True


def create_training_instances(files):
    kto_data_dic = {'text':[]}
    unique_prompts = []
    for jj, tree in enumerate(files):
        for training_example, success, _ in tree:
            if len(training_example)==0:
                continue
            if training_example in unique_prompts:
                continue
            unique_prompts.append(training_example)
            if training_example[-1]['role']=='user':
                training_example = training_example[:-1]
            for idx, dic in enumerate(reversed(training_example)):
                if dic.get('role')=='user':
                    break
                if not dic.get('content').startswith('APIRETURN'):
                    final_label = success
                    if success:
                        resulting_utt = None
                        if -idx < 0:
                            resulting_utt = training_example[-idx]
                        final_label = clean_good_utterance(dic['content'], resulting_utt = resulting_utt)
                        if final_label:
                            kto_data_dic['text'].append(tokenizer.apply_chat_template([{'role':'system', 'content':MONO_PROMPT}] + training_example[:len(training_example)-idx], add_generation_prompt=True, tokenize=False))
                    
    return kto_data_dic

dataset = create_training_instances(files)

my_dataset = DatasetDict({'train':Dataset.from_pandas(pd.DataFrame(dataset)).shuffle(seed=42)})


from peft import LoraConfig, get_peft_model


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_dropout=0.05,
    bias="none",
)
model.enable_input_require_grads()
model = get_peft_model(model, config).to('cuda')

from trl import SFTTrainer
from transformers import TrainingArguments
import logging

logging.basicConfig(
    level=logging.INFO, # Log at INFO level or higher
    format="%(asctime)s [%(levelname)s] %(message)s", # Include timestamp in logs
    handlers=[
        logging.FileHandler("/home/blattimer/code/multiwoz-api/logs/tlog.log"), # Log to a file
        logging.StreamHandler() # Log to console
    ]
)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = my_dataset["train"],
    dataset_text_field = "text",
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    max_seq_length = 8264,
    args = TrainingArguments(
        num_train_epochs = 2,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        logging_dir = "/home/blattimer/code/multiwoz-api/logs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "data/sft_0603_output",
        gradient_checkpointing = True,
        save_strategy= "steps",
        save_steps= 67,
        report_to='tensorboard',
    ),
)
trainer_stats = trainer.train()
