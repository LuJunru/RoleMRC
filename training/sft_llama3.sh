export GLOO_SOCKET_IFNAME=eth0
export GLOO_SOCKET_TIMEOUT=300
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1

export GPU_NUM_PER_NODE=4
export WORLD_SIZE=1

MAXLEN=4200
EPOCH=1
SAVEINTERVAL=${EPOCH}
SEED=42
PER_GPU_BATCH=8
GRA_ACC=2
LR=2e-5

raw_model_path=/raw_model_path/llama3_1_8B/
train_data_path=./data/roleMRC_train-sft_mix.jsonl
deepspeed_config_path=./ds_config.json
model_output_path=/output_path/llama3_1_8B-SFT-mix/
cache_path=./cache
resume_path=${model_output_path}/None

TRAINDATANUM=$(wc -l <"${train_data_path}")
SAVESTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * ${SAVEINTERVAL} * $WORLD_SIZE)) + 1}")
TOTALSTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * $WORLD_SIZE)) + 1}")
echo "We use $WORLD_SIZE nodes to train with ${TRAINDATANUM} samples for ${EPOCH} epochs, resulting in ${TOTALSTEP} running steps, and thus we will save checkpoints every ${SAVESTEP} steps."

# training
export CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nnodes=$WORLD_SIZE \
    --node_rank=0 \
    --nproc_per_node $GPU_NUM_PER_NODE \
    train_sft.py \
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
    --learning_rate ${LR} \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --deepspeed ${deepspeed_config_path} \
    --tf32 True \
    --model_max_length ${MAXLEN} \
    --train_data_path ${train_data_path} \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --gradient_checkpointing True \
    --cache_dir ${cache_path} \
    --resume_from_checkpoint ${resume_path} \
    --seed ${SEED} \
    --report_to "none"
