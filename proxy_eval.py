import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOVanillaTrainer, set_seed
from trl.core import LengthSampler
import numpy as np
import pandas as pd
tqdm.pandas()
from peft import LoraConfig

import pandas as pd

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}
root_dir = "/home/winnie"
rm_dir = f"{root_dir}/output_models/rms"
# eval_data_path = "/apdcephfs/private_radyang/trl/examples/rlhf/data/eval_prompt.json"
# eval_dataset = build_dataset(config, tokenizer, eval_data_path)
rm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B") #TODO: remove hard code

# files = ["rm_2sft_full_train_1epoch_llama_13b/merged_rm", "rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp5", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp6", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp7", ] # TODO: Currently hardcoded bc of llama
# files = ["rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp5", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp6", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp7", ] # TODO: Currently hardcoded bc of llama
files = [f"0715_relabel_rm_gpt_neo_2_7b_5e-6_1epoch_{i}" for i in range(1,7)]
k = len(files)
print(k)

means_rms = torch.tensor([1.8175465893000364,1.9616623278707266,1.91334992274642,0.6972120497375727,1.1580963432788849,2.7236480712890625])
stds_rms = torch.tensor([1.1722500539476755,1.2413885538391234,1.1708371368464998,1.308637960323578,1.3141790084902882,1.1942363257298647])

reward_record = []

def clean_text(text):
    stext = [x for x in text.split("###") if x]
    return (stext[0].strip("#")).rstrip()

# def evaluate(dataset, batch_num):
#     eval_bs = 1
#     batch = {}
#     dataset_size = len(dataset)
#     response_tensors = []
#     for i in tqdm(range(dataset_size // eval_bs)):
#         query_tensor = dataset[:dataset_size]["input_ids"][i*eval_bs: (i+1)*eval_bs]

#         with torch.no_grad():
#             response_tensor = ppo_trainer.generate(
#                 query_tensor, return_prompt=False, length_sampler=output_length_sampler, 
#                 batch_size=eval_bs, **generation_kwargs
#             )
#         response_tensors.extend(response_tensor)
    
#     full_responses = dataset[f"batch{batch_num}/response"]
#     clean_texts = [clean_text(tmp_text) for tmp_text in full_responses]
#     batch['query'] = dataset['query']
#     batch['response'] = clean_texts 
#     texts_for_rewards = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts_for_rewards, **sent_kwargs)
#     rewards = [output[0]["score"] for output in pipe_outputs]
#     batch['score'] = rewards
#     return batch

# full_responses = tokenizer.batch_decode(response_tensors)
# clean_texts = [clean_text(tmp_text) for tmp_text in full_responses]
# clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]




# lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
# response_tensors = [response_tensors[j][:np.max([lengths[j], 1])] for j in range(len(response_tensors))]
process_id = Accelerator().local_process_index
x2 = [15, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32]
rgs = [(15,18), (18,21), (21,24), (24,27), (27,33)]

df = pd.read_csv(f"/home/winnie/coef2.0_testFalse_num{rgs[0][0]}_{rgs[0][1]}_8bit_tmp.csv", index_col=0)
i = 1
for rg in rgs[1:]:
    curr = pd.read_csv(f"/home/winnie/coef2.0_testFalse_num{rg[0]}_{rg[1]}_8bit_tmp.csv", index_col=0)
    curr = curr.drop("query", axis=1)
    df = pd.concat([df, curr], axis=1)

res = pd.DataFrame()
for batch_i in x2:

    # Compute score
    texts_for_rewards = [q + r for q, r in zip(df["query"], df[f"batch{batch_i}/response"])]
    rewards = torch.zeros(k, len(texts_for_rewards))
    sentiment_pipes = []

    # for filename in os.listdir(rm_dir):
    for i, filename in enumerate(files):
        print(f"Process {process_id}, Loading model {i}")
        f = os.path.join(rm_dir, filename)
        pipe = pipeline("sentiment-analysis", model=f, device=process_id, tokenizer=rm_tokenizer)
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        pipe_outputs = pipe(texts_for_rewards, **sent_kwargs)
        # rewards[i,:] = (torch.tensor([output[0]["score"] for output in pipe_outputs]) - rms_mean[i]) / rms_std[i]
        rewards[i,:] = torch.tensor([output[0]["score"] for output in pipe_outputs])
    pipe = None # Free memory (?)

    # Normalize rewards
    print(rewards)
    rewards = (rewards - means_rms.reshape(k, 1).expand(rewards.shape)) / stds_rms.reshape(k, 1).expand(rewards.shape)
    rewards_mean = rewards.mean(axis=0)
    print(rewards_mean)
    rewards_std = rewards.std(axis=0)

    res[f"batch{batch_i}/rms_mean"] = rewards_mean
    res[f"batch{batch_i}/rms_std"] = rewards_std
    print(res)

res.to_csv("coef2.0_rms_eval_0_32.csv")

    # batch_rewards_mean = rewards_mean.float().mean().item()
    # print("process {}, iter {}, batch {}: mean score: {}, std: {}".format(process_id, epoch, batch_i, batch_rewards_mean, rewards_std.mean().item()))
    # reward_record.append(batch_rewards_mean)