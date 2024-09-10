import json
import os
from datasets import Dataset, DatasetDict 
import pandas as pd
from trl import KTOConfig, KTOTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os
from datasets import load_dataset
from peft import LoraConfig
config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_dropout=0.05,
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
)
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

model_ref = AutoModelForCausalLM.from_pretrained(
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


dataset = load_dataset("json", data_files="/root/code/multiwoz-api/train_data_kto_just_div.jsonl", split="train")

def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example

formatted_dataset = dataset.map(format_dataset)

training_args = KTOConfig(
        report_to="tensorboard",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-7,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        logging_dir="/root/code/multiwoz-api/logs/",
        optim = "adamw_8bit",
        lr_scheduler_type = "cosine",
        seed = 42,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=2.71,
        gradient_checkpointing = True,
        save_strategy= "steps",
        save_steps= 100,
        max_prompt_length=5000,
        max_length=5256,
        output_dir = 'data/llama_8b_kto_0612',
    )

kto_trainer = KTOTrainer(
    model = model,
    ref_model = model_ref,
    peft_config = config,
    args = training_args,
    train_dataset = formatted_dataset,
    tokenizer = tokenizer,
)


kto_trainer.train()
