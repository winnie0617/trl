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
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import numpy as np
import pandas as pd
tqdm.pandas()
from peft import LoraConfig



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
    model_name: Optional[str] = field(default='/home/xiongwei/LMFlow/output_models/clean_llama_7b_sft_1epoch', metadata={"help": "the model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='/home/yangrui/wchow/trl/logs_trl/')
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    target: Optional[float] = field(default=0.5, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.0,metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=1, metadata={"help": "Maximum gradient norm for gradient clipping"})
    wandb_name: Optional[str] = field(default='ppo_llama_klcoef0', metadata={"help": "Name for this experiment"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    target=script_args.target,
    max_grad_norm=script_args.max_grad_norm,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    wandb_name=script_args.wandb_name
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 20}

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, tokenizer, dataset_name=''):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    ds = load_dataset("json", data_files=dataset_name, split="train")['instances'][0]
    texts = [sample['text'] for sample in ds]

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


# We retrieve the dataloader by calling the `build_dataset` function.
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)

# Now let's build the model, the reference model, and the tokenizer.
# model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

process_id = Accelerator().local_process_index
gpu_id_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}
pipeline_gpu_id_map = {
    0: 4,
    1: 5,
    2: 6,
    3: 7
}

gpu_id = gpu_id_map[process_id]
pipeline_gpu_id = pipeline_gpu_id_map[process_id]
print('process: {}, model gpu id: {}, pipline gpu id: {}'.format(process_id, gpu_id, pipeline_gpu_id))

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    peft_config=lora_config,
    device_map=gpu_id,
)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    load_in_8bit=True, 
    peft_config=lora_config,
    device_map=gpu_id,
    )

tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast = False)

tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN,
    }
)


data_path = "/apdcephfs/private_radyang/trl/examples/rlhf/data/train_prompt.json"
dataset = build_dataset(config, tokenizer, data_path)
# eval_data_path = "/apdcephfs/private_radyang/trl/examples/rlhf/data/eval_prompt.json"
# eval_dataset = build_dataset(config, tokenizer, eval_data_path)
rm_tokenizer = AutoTokenizer.from_pretrained("/apdcephfs/private_radyang/trl/examples/rlhf/data/rewardmodels/rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4")


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config, model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)
device = ppo_trainer.accelerator.device

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="/apdcephfs/private_radyang/trl/examples/rlhf/data/rewardmodels/rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4",
    device=pipeline_gpu_id,
    # device='auto'
    tokenizer=rm_tokenizer
    )




# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,  
    # "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 96
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)
reward_record = []

def clean_text(text):
    stext = [x for x in text.split("###Human") if x]
    return (stext[0].strip("#")).rstrip()


model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True
epochs = script_args.epochs
for epoch in range(epochs):
    for i, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
            )

        full_responses = tokenizer.batch_decode(response_tensors)
        clean_texts = [clean_text(tmp_text) for tmp_text in full_responses]
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]

        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        response_tensors = [response_tensors[j][:np.max([lengths[j], 1])] for j in range(len(response_tensors))]
        batch['response'] = clean_texts

     
        texts_for_rewards = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts_for_rewards, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        print("iter {}, batch {}: mean score: {}".format(epoch, i, np.mean(rewards)))
        reward_record.append(np.mean(rewards))

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print("iter {}, batch {}: log finish".format(epoch, i))


    # save model
    save_path = os.path.join(config.save_directory, config.wandb_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)