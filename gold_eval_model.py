# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import get_peft_model_state_dict

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, PPOVanillaTrainer, set_seed
from trl.core import LengthSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import time

tqdm.pandas()
from peft import LoraConfig
# import matplotlib.pyplot as plt


########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # reference model
    model_name: Optional[str] = field(default='/home/winnie/output_models/0715_relabel_sft_llama_7b_2e-5_1epoch', metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    # Model to be evaluated
    peft_model_path: Optional[str] = field(default='/home/winnie/trl/logs_trl/ppo_llamavanilla_gradaccu1_gradnorm1_bs2048_coef2.0')
    # Gold model:
    reward_model_path: Optional[str] = field(default='/home/winnie/output_models/0715_openllama_13b_gold_rm_2sft_lora16_3e-5_1epoch_1x8x2bs/merged_rm_715')
    test_data: Optional[bool] = field(default=False)

size = 1024 # size of dataset to use (might be smaller than this because some samples are too long)
# models on gpu 0,1,2,3 gold on gpu 4,5,6,7
num_processes = torch.cuda.device_count()
current_device = Accelerator().local_process_index
model_gpu = current_device
# reference_gpu = 1
reward_gpu = current_device + (num_processes // 2)
reward_gpu = 1

print(f"process: {current_device}, gold rm on {reward_gpu}, models on {model_gpu}")

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
config = PPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    optimize_cuda_cache=True,
)
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

def build_dataset(tokenizer, dataset_name='', size=size):
    ds = load_dataset("json", data_files=dataset_name, split="train")['instances'][0]
    texts = [sample['text'] for sample in ds][:size]
    ds = Dataset.from_dict({
        "text": texts,
    })

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])[:]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(8888)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tokenizer = AutoTokenizer.from_pretrained(script_args.peft_model_path + '/batch_0', use_fast = False)
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN,
    }
)
tokenizer.padding_side = "left"

# ref_model = AutoModelForCausalLM.from_pretrained(
#         script_args.model_name,
#         load_in_8bit=True, 
#         peft_config=lora_config,
#         # torch_dtype=torch.float16,
#         device_map=reference_gpu,
#     )

if script_args.test_data:
    data_path = '/home/winnie/data/clean_hh_rlhf_uncerrtainty_study/rlhf_eval_prompt/clean_hh_rlhf_rlhf_eval_prompt.json'
else:
    data_path = '/home/winnie/data/clean_hh_rlhf_uncerrtainty_study/rlhf_train_prompt/clean_hh_rlhf_rlhf_prompt.json'

print('test data: ', script_args.test_data)
print(data_path)

train_dataset = build_dataset(tokenizer, data_path)
print(train_dataset)

# Load Gold Model

# from transformers import AutoModelForSequenceClassification
# ref_model = AutoModelForSequenceClassification.from_pretrained(
#         script_args.reward_model_path,
#         # load_in_8bit=True, 
#         # peft_config=lora_config,
#         torch_dtype=torch.bfloat16,
#         device_map=1,
#     )

rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_path)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=script_args.reward_model_path,
    device=reward_gpu,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
    # device='auto'
    )

# Prepare reference model

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9999,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,  
    # "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 96
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def clean_text(text):
    stext = [x for x in text.split("###Human") if x]
    return (stext[0].strip("#")).rstrip()


def evaluate(ppo_trainer, dataset):
    eval_bs = 1
    batch = {}
    dataset_size = len(dataset)
    response_tensors = []
    for i in tqdm(range(dataset_size // eval_bs)):
        query_tensor = dataset[:dataset_size]["input_ids"][i*eval_bs: (i+1)*eval_bs]

        with torch.no_grad():
            response_tensor = ppo_trainer.generate(
                query_tensor, return_prompt=False, length_sampler=output_length_sampler, 
                batch_size=eval_bs, **generation_kwargs
            )
        response_tensors.extend(response_tensor)
    
    full_responses = tokenizer.batch_decode(response_tensors)
    clean_texts = [clean_text(tmp_text) for tmp_text in full_responses]
    batch['query'] = dataset[:(i+1)*eval_bs]['query']
    batch['response'] = clean_texts 
    texts_for_rewards = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts_for_rewards, **sent_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    batch['score'] = rewards
    return batch

### REFERENCE MODEL, NO NEED
# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#         script_args.model_name,
#         load_in_8bit=True,
#         torch_dtype=torch.bfloat16, 
#         peft_config=lora_config,
#         device_map=model_gpu,
#     )
# config.model_name = script_args.model_name
# ppo_trainer = PPOVanillaTrainer(config, model, tokenizer=tokenizer)
# train_refmodel_res = evaluate(ppo_trainer, train_dataset)
# print(np.mean(train_refmodel_res['score']))
# df_results = pd.DataFrame(train_refmodel_res)
# df_results.rename(columns={"response": "reference/response", "score": "reference/score"})
# import gc 
# del model
# del ppo_trainer
# gc.collect()
# torch.cuda.empty_cache()

# Prepare models to be evaluated



df_results = pd.DataFrame()
peft_model_num_start = 27
peft_model_num_end = 33
batch_nums = [28, 30, 32]
# peft_model_num_start = (6*current_device)
# peft_model_num_end = min(6+6*current_device, 24)
print(f"Device {current_device} - start={peft_model_num_start}, end={peft_model_num_end}")
for k in batch_nums:
    peft_model_path = os.path.join(script_args.peft_model_path, 'batch_{}'.format(k))
    print('++++++++++++++++++++++++')
    print('loading model {} from {}'.format(k, peft_model_path))
    print('++++++++++++++++++++++++')
    # config originally uses ref model
    config.model_name = peft_model_path

    peft_config = PeftConfig.from_pretrained(peft_model_path)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        peft_model_path,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16, 
        peft_config=lora_config,
        device_map=model_gpu,
    )
    ppo_trainer = PPOVanillaTrainer(config, model, tokenizer=tokenizer)

    # model = AutoModelForCausalLM.from_pretrained(
    #     peft_model_path,
    #     # torch_dtype=torch.bfloat16, 
    #     load_in_8bit=True, 
    #     device_map=model_gpu
    # )
    # model = PeftModel.from_pretrained(model, peft_model_path)
    # model.eval()

    train_model_res = evaluate(ppo_trainer, train_dataset)
    df_results['query'] = train_model_res['query']
    df_results['batch{}/response'.format(k)] = train_model_res['response']
    df_results['batch{}/score'.format(k)] = train_model_res['score']
    print(np.mean(train_model_res['score']))
    df_results.to_csv('coef2.0_test{}_num{}_{}_8bit_tmp.csv'.format(script_args.test_data, peft_model_num_start, peft_model_num_end))

    ## del ppo_trainer, model, ref_model
    import gc 
    del model
    del ppo_trainer
    gc.collect()
    torch.cuda.empty_cache()

df_results.to_csv(datetime.now().strftime('coef2.0_test{}_num{}_{}_%Y_%m_%d_%H_%M.csv'.format(script_args.test_data, peft_model_num_start, peft_model_num_end)))



################3
# load adapter
    # model = model.merge_and_unload()
    
    # adapter_weights = torch.load(peft_model_path + '/adapter_model.bin')
    # adapter_weights_rename = {}
    # for key, value in adapter_weights.items():
    #     new_name_lis = key.split('.')
    #     new_name_lis.insert(-1, 'batch_{}'.format(k))
    #     new_name = '.'.join(new_name_lis)
    #     adapter_weights_rename[new_name] = value
   
    # current_state_dict = model.base_model.model.pretrained_model.state_dict()
    # current_state_dict.update(adapter_weights_rename)
    # import pdb;pdb.set_trace()
    # model.base_model.model.pretrained_model.load_state_dict(current_state_dict)
    # model.set_adapter('batch_{}'.format(k))
    # model.merge_adapter()