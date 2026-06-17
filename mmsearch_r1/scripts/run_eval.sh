#!/bin/bash
# 通用评估脚本: 评估任意 checkpoint
#
# 运行: bash mmsearch_r1/scripts/run_eval.sh checkpoints/mmsearch-r1/grpo-7b-fvqa-penalty0.1-balanced
#       bash mmsearch_r1/scripts/run_eval.sh Qwen/Qwen2.5-VL-7B-Instruct

# ====== 环境变量 ======
export VAL_DATA_PATH="/root/autodl-tmp/multimodal-search-r1/data/FVQA/fvqa_test.parquet"
export WANDB_PROJECT_NAME="mmsearch-r1"

# ====== 实验配置 ======
MODEL_PATH=${1:-Qwen/Qwen2.5-VL-7B-Instruct}
EXP_NAME=$(basename $MODEL_PATH)
WANDB_EXP_NAME="eval_${EXP_NAME}"
N_GPUS=3
SAVE_DIR="${MODEL_PATH}/eval_results"

mkdir -p "$SAVE_DIR"
python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.user_prompt_round_1=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
    data.user_prompt_after_image_search=mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl \
    data.user_prompt_after_text_search=mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmsearch \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_gen_round=3 \
    actor_rollout_ref.rollout.response_length_total=8192 \
    actor_rollout_ref.rollout.search.topk=5 \
    actor_rollout_ref.rollout.search.image_search_limit=1 \
    actor_rollout_ref.rollout.search.text_search_limit=2 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$SAVE_DIR \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.val_only_save_dir=$SAVE_DIR \
    trainer.val_generations_to_log_to_wandb=64 \
    2>&1 | tee "${SAVE_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
