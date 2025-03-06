export GLOO_SOCKET_IFNAME=eth0
export GLOO_SOCKET_TIMEOUT=300
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1

# pip3 install --upgrade peft==0.5.0 datasets transformers==4.44.0 typing-extensions==4.11.0 accelerate==0.30.1 deepspeed==0.15.4

MAXLEN=$1
EPOCH=1
SAVEINTERVAL=1
train_data_file=$2
postfix=$3
SEED=42
export GPU_NUM_PER_NODE=4
export WORLD_SIZE=1

pgb=$4
acc=$5
ds_file=$6

model=$7
# raw_model_path=Qwen/Qwen2.5-7B
raw_model_path=meta-llama/Llama-3.1-8B
train_data_path=./data/${train_data_file}
deepspeed_config_path=./ds_config.json
model_output_path=/output_path/llama3_1_${model}-${postfix}-mix/
cache_path=./cache
resume_path=${model_output_path}/$8

case ${model} in 
    "8B")
        PER_GPU_BATCH=${pgb}
        GRA_ACC=${acc}
        LR=2e-5
        ;;
    "7B")
        PER_GPU_BATCH=${pgb}
        GRA_ACC=${acc}
        LR=2e-5
        ;;
    "13B")
        PER_GPU_BATCH=${pgb}
        GRA_ACC=${acc}
        LR=5e-6
        ;;
    "70B")
        PER_GPU_BATCH=${pgb}
        GRA_ACC=${acc}
        LR=5e-6
        ;;
esac

TRAINDATANUM=$(wc -l <"${train_data_path}")
SAVESTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * ${SAVEINTERVAL} * $WORLD_SIZE)) + 1}")
TOTALSTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * $WORLD_SIZE)) + 1}")
# EVALSTEP=100
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
