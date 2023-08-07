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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import numpy as np

# import evaluate
# from gem_metrics.msttr import MSTTR
# from gem_metrics.ngrams import NGramStats
# from gem_metrics.texts import Predictions
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

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="/home/winnie/output_models/sft_llama_7b_2e-5_1epoch",
                                      metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=3, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.05, metadata={"help": "kl target for early stopping"})


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
    optimize_cuda_cache=True,
    init_kl_coef=0.001
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, tokenizer, dataset_name="/home/winnie/data/clean_hh_rlhf_uncerrtainty_study/rlhf_train_prompt/clean_hh_rlhf_rlhf_prompt.json"):
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
    # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    # ds = load_dataset(dataset_name, split="train")
    # ds = ds.rename_columns({"text": "review"})
    # ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    ds = load_dataset("json", data_files=dataset_name, split="train")['instances'][0]
    texts = [sample['text'] for sample in ds]
    # ref_texts = [sample['ref_text'] for sample in ds]
    # intents = [sample['intent'] for sample in ds]
    from datasets import Dataset
    ds = Dataset.from_dict({
        "text": texts,
        # "ref_text":ref_texts,
        # "intent": intents
    })

    # print(ds.col)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])[:]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)
    ds.set_format(type="torch")
    print(len(ds))
    print(ds.column_names)
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index

# Now let's build the model, the reference model, and the tokenizer.
# model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# print(current_device)
# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    # load_in_8bit=True,
    peft_config=lora_config,
    # layer_norm_names=[],
    # device_map={"": current_device},
)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
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
dataset = build_dataset(config, tokenizer)
##############
rm_path = "weqweasdas/hh_rlhf_rm_open_llama_3b"
rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
# rm_tokenizer.pad_token = rm_tokenizer.eos_token
# rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id #tokenizer.eos_token
# rm_tokenizer.padding_side = "left"
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=rm_path,
    model_kwargs={"load_in_8bit": True},
    # device_map={"": current_device},
    # device=device)
)
# tokenizer=rm_tokenizer)

# sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
print("Using device", device)


# meteor = evaluate.load('meteor')



# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,  # "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 96
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

reward_record = []
store = 0


def clean_text(text):
    stext = [x for x in text.split("###Human") if x]
    return (stext[0].strip("#")).rstrip()


# model.gradient_checkpointing_enable()
# model.pretrained_model.config.use_cache = False

model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    with torch.no_grad():
        response_tensors = ppo_trainer.generate(
            query_tensors, batch_size=14, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
        )
    #    print(response_tensors.shape)
    '''
    for i in range(5):
        print(len(response_tensors[i]))
        print(response_tensors[i])

    clean_response_tensors = [res_tensor[:9] for res_tensor in response_tensors]


    for i in range(5):
        print(len(clean_response_tensors[i]))
        print(clean_response_tensors[i])
    '''

    full_responses = tokenizer.batch_decode(response_tensors)
    clean_texts = [clean_text(tmp_text) for tmp_text in full_responses]
    clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
    lengths = [len(clean_tensor) for clean_tensor in clean_response_tensors]

    # my_device = query_tensors[0].device
    # response_tensors = [(response_tensors[i][:np.max([lengths[i]-1, 1])]).to(my_device) for i in range(len(response_tensors))]

    response_tensors = [response_tensors[i][:np.max([lengths[i] - 2, 1])] for i in range(len(response_tensors))]

    batch["response"] = clean_texts

    # test_texts = [tokenizer.decode(tmp_tensor) for tmp_tensor in response_tensors]

    for i in range(1):
        print(clean_texts[i])
        # print(test_texts[i], len(test_texts[i]), " | ", clean_texts[i], len(clean_texts[i]))
        # print(response_tensors[i], len(response_tensors[i]), "|", clean_response_tensors[i], len(clean_response_tensors[i]))
    # Compute sentiment score
    texts = batch["response"]
    # ref_texts = batch["ref_text"]
    # intents = batch['intent']

    # rewards = get_rewards(texts, ref_texts, intents)
    '''
    rewards = []
    for jj in range(8):
        tmp_rewards = get_rewards(texts[jj*16 : (jj+1) * 16], ref_texts[jj*16 : (jj+1) * 16], intents[jj*16 : (jj+1) * 16])
        rewards.extend(tmp_rewards)
    assert len(rewards) == 128
    '''

    # clean_texts = [clean_text(tmp_text) for tmp_text in tmp_responses]
    # rewards = get_rewards(clean_texts, ref_texts, intents)
    texts_for_rewards = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts_for_rewards, **sent_kwargs)
    # rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

    print("iter " + str(epoch), np.mean(rewards))
    reward_record.append(np.mean(rewards))

    # reward_record.append(np.mean(rewards))
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    # model.gradient_checkpointing_enable()
    # model.pretrained_model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    # print(query_tensors[0].device, response_tensors[0].device, rewards[0].device)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # eval_model(batch["response"], model, tokenizer)

    import matplotlib.pyplot as plt

    plt.plot(reward_record)
    plt.savefig('hh_rlhf_ppo/new_reward_record_' + str(ppo_trainer.accelerator.device) + '.png')
    np.save('hh_rlhf_pponew_reward_record_' + str(ppo_trainer.accelerator.device) + '.npy',
            reward_record)

    if False:
        model.save_pretrained('/root/data/weixiong/output_models/hh_rlhf_ppo/llama' + str(store), push_to_hub=False)
        tokenizer.save_pretrained('/root/data/weixiong/output_models/hh_rlhf_ppo/llama' + str(store), push_to_hub=False)
        store += 1