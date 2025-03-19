export GLOO_SOCKET_IFNAME=eth0
export GLOO_SOCKET_TIMEOUT=300
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1

export GPU_NUM_PER_NODE=4
export WORLD_SIZE=1

MAXLEN=3000 # default 4096
EPOCH=1
SAVEINTERVAL=${EPOCH}
LR=$(awk "BEGIN {print 4e-7 * sqrt($WORLD_SIZE)}")
BETA=0.1
LR_TYPE=linear
SEED=42
RLType=sigmoid
ISLORA=0  # 1 for lora, 0 for full
M_TYPE=llama  # default llama, llama / qwen2
PER_GPU_BATCH=4
GRA_ACC=4

raw_model_path=/output_path/llama3_1_8B-SFT-mix/
train_data_path=./data/roleMRC_train-rl_mix.jsonl
eval_data_path=./data/roleMRC_dev-rl.jsonl
deepspeed_config_path=./ds_config.json
model_output_path=/output_path/${RLType}_llama3_1_8B-DPO-mix/
cache_path=./cache

TRAINDATANUM=$(wc -l < "${train_data_path}")
SAVESTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * ${SAVEINTERVAL} * $WORLD_SIZE)) + 1}")
TOTALSTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * $WORLD_SIZE)) + 1}")
EVALSTEP=100

echo "We use $WORLD_SIZE nodes to train with ${TRAINDATANUM} samples for ${EPOCH} epochs, resulting in ${TOTALSTEP} running steps, and thus we will save checkpoints every ${SAVESTEP} steps."

# training
export CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nnodes=$WORLD_SIZE \
    --node_rank=0 \
    --nproc_per_node $GPU_NUM_PER_NODE \
    dpo_tuning_demo.py \
    --model_name_or_path ${raw_model_path} \
    --bf16 True \
    --output_dir ${model_output_path} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${PER_GPU_BATCH} \
    --gradient_accumulation_steps ${GRA_ACC} \
    --save_strategy "steps" \
    --save_steps ${SAVESTEP} \
    --save_total_limit 1 \
    --per_device_eval_batch_size ${PER_GPU_BATCH} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVALSTEP} \
    --learning_rate ${LR} \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type ${LR_TYPE} \
    --deepspeed ${deepspeed_config_path} \
    --tf32 True \
    --model_max_length ${MAXLEN} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --gradient_checkpointing True \
    --report_to "none" \
    --loss_type ${RLType} \
    --if_lora ${ISLORA} \
    --dpo_beta ${BETA} \
    --cache_dir ${cache_path} \
    --seed ${SEED}
