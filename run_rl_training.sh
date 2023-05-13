#!/bin/sh

python examples/stack_llama/scripts/rl_training.py \
--model_name decapoda-research/llama-7b-hf \
--tokenizer_name decapoda-research/llama-7b-hf \
--reward_model_name decapoda-research/llama-7b-hf \
--log_with wandb \
--batch_size 1 \


