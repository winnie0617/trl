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

# root_dir = "/home/winnie"
root_dir = "/home/yangrui/wchow"
print("COUNT", torch.cuda.device_count())

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # model_name: Optional[str] = field(default='/apdcephfs/private_radyang/trl/examples/rlhf/data/hh_rlhf_llama_7b_sft_on_relabel_set_blocksize_1024', metadata={"help": "the model name"})
    model_name: Optional[str] = field(default=f'{root_dir}/output_models/0715_relabel_sft_llama_7b_2e-5_1epoch', metadata={"help": "the model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default=f'{root_dir}/trl/logs_trl/')    
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    target: Optional[float] = field(default=3, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.01,metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=1, metadata={"help": "Maximum gradient norm for gradient clipping"})
    wandb_name: Optional[str] = field(default='ppo_llamavanilla_gradaccu1_gradnorm1_bs2048_clean###', metadata={"help": "Name for this experiment"})
    uncertainty_coef: Optional[float] = field(default=0.5,metadata={"help": "Uncertainty penalty coefficient"},)


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
    wandb_name=f"ppo_llamavanilla_gradaccu1_gradnorm1_bs{script_args.batch_size * torch.cuda.device_count()}_coef{script_args.uncertainty_coef}",
    save_directory=script_args.save_directory,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}


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
print("Accelerator local_process_index", current_device)

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
# gpu_id_map = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3
# }
# pipeline_gpu_id_map = {
#     0: 4,
#     1: 5,
#     2: 6,
#     3: 7
# }

# gpu_id = gpu_id_map[process_id]
# pipeline_gpu_id = pipeline_gpu_id_map[process_id]

gpu_id = process_id
# pipeline_gpu_id_start = process_id + 1
# print('process: {}, model gpu id: {}, pipline gpu id: {}'.format(process_id, gpu_id, pipeline_gpu_id_start))
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))



model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    peft_config=lora_config,
    device_map=process_id,
)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast = False)

tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN,
    }
)

############
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = "left"

rm_dir = f"{root_dir}/output_models/rms"
# rm_path = "/home/share/rewardmodels/rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4"
data_path = f"{root_dir}/data/clean_hh_rlhf_uncerrtainty_study/rlhf_train_prompt/clean_hh_rlhf_rlhf_prompt.json"
dataset = build_dataset(config, tokenizer, data_path)
# eval_data_path = "/apdcephfs/private_radyang/trl/examples/rlhf/data/eval_prompt.json"
# eval_dataset = build_dataset(config, tokenizer, eval_data_path)
rm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B") #TODO: remove hard code

# files = ["rm_2sft_full_train_1epoch_llama_13b/merged_rm", "rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp5", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp6", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp7", ] # TODO: Currently hardcoded bc of llama
# files = ["rm_2sft_full_train_1epoch_gpt_neo_2_7B_exp4", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp5", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp6", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp7", ] # TODO: Currently hardcoded bc of llama
files = [f"0715_relabel_rm_gpt_neo_2_7b_5e-6_1epoch_{i}" for i in range(1,7)]
k = len(files)

means_rms = torch.tensor([1.8175465893000364,1.9616623278707266,1.91334992274642,0.6972120497375727,1.1580963432788849,2.7236480712890625])
stds_rms = torch.tensor([1.1722500539476755,1.2413885538391234,1.1708371368464998,1.308637960323578,1.3141790084902882,1.1942363257298647])

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOVanillaTrainer(
    config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)
print("PPO Trainer Number Processes", ppo_trainer.accelerator.num_processes)
print("Total Batch Size", ppo_trainer.accelerator.num_processes * config.batch_size)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device

# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=rm_path,
#     device=pipeline_gpu_id,
#     # device='auto'
#     tokenizer=rm_tokenizer
#     )



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
output_min_length = 196
output_max_length = 256
output_length_sampler = LengthSampler(output_min_length, output_max_length)
reward_record = []

def clean_text(text):
    stext = [x for x in text.split("###") if x]
    return (stext[0].strip("#")).rstrip()

model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True
epochs = script_args.epochs
for epoch in range(epochs):
    for batch_i, batch in enumerate(ppo_trainer.dataloader):
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

        # Compute score
        texts_for_rewards = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = torch.zeros(k, config.batch_size)

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
        print("Before norm")
        print(rewards)
        rewards = (rewards - means_rms.reshape(k, 1).expand(rewards.shape)) / stds_rms.reshape(k, 1).expand(rewards.shape)
        print("after norm")
        print(rewards)

        rewards_mean = rewards.mean(axis=0).to(ppo_trainer.current_device)
        rewards_std = rewards.std(axis=0).to(ppo_trainer.current_device)

        penalized_rewards = rewards_mean - script_args.uncertainty_coef * rewards_std # TODO: remove hard-coding

        batch_rewards_mean = rewards_mean.float().mean().item()
        print("process {}, iter {}, batch {}: mean score: {}, std: {}".format(process_id, epoch, batch_i, batch_rewards_mean, rewards_std.mean().item()))
        reward_record.append(batch_rewards_mean)

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, [torch.tensor(r) for r in penalized_rewards]) # rewards must be a list of tensors
        ppo_trainer.log_stats(stats, batch, rewards_mean, stds=rewards_std) 
        print("process {}, iter {}, batch {}: log finish".format(process_id, epoch, batch_i))


        # save model
        if process_id == 0:
            save_path = os.path.join(config.save_directory, config.wandb_name, 'batch_{}'.format(batch_i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, batch_i))
