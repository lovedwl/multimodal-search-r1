#!/bin/bash
# SFT 训练: 用 GPT-5.5 蒸馏的数据微调 Qwen2.5-VL-7B
# 输出: checkpoints/mmsearch-r1/sft_tool_classifier

# ====== 环境变量 ======
export WANDB_PROJECT_NAME="mmsearch-r1"

# ====== 实验配置 ======
WANDB_EXP_NAME="sft_tool_classifier"
N_GPUS=3
SFT_DATA="/root/autodl-tmp/multimodal-search-r1/data/FVQA/sft_train.parquet"

mkdir -p "checkpoints/mmsearch-r1/${WANDB_EXP_NAME}"
python3 -m verl.trainer.sft_trainer \
    data.train_files=$SFT_DATA \
    data.messages_key=messages \
    data.image_key=images \
    data.train_batch_size=24 \
    data.micro_batch_size_per_gpu=8 \
    data.max_token_len_per_gpu=8192 \
    data.use_dynamic_bsz=True \
    data.truncation=right \
    data.max_length=4096 \
    model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.default_local_dir=checkpoints/mmsearch-r1/$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.logger=['console','wandb'] \
    trainer.seed=42 \
    2>&1 | tee "checkpoints/mmsearch-r1/${WANDB_EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"
