import json
import os
from datasets import Dataset, DatasetDict 
import pandas as pd
from trl import KTOConfig, KTOTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import numpy as np
import os
from josh_train.utils import parse_api_call
from huggingface_hub import login
login(token="HF_TOKEN")
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
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


dataset = load_dataset("json", data_files="/root/code/multiwoz-api/train_data_full_convo.jsonl", split="train")
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    peft_config = config,
    args = TrainingArguments(
        num_train_epochs = 3,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        learning_rate = 2e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        logging_dir = "/root/code/multiwoz-api/multiwoz_api/logs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "data/sft_0611_output",
        gradient_checkpointing = True,
        save_strategy= "steps",
        save_steps= 13,
        report_to='tensorboard',
    ),
)
trainer_stats = trainer.train()
