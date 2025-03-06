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
DPOP_LAMBADA=50.0
SFT_LAMBADA=1.0
L1_LAMBADA=0.01
TDPO_ALPHA=0.5
NEFTALPHA=5.0
SIMPO_LAMBADA=1.4
LR_TYPE=linear

NOISE=$1  # 0 for no noise and 1 for noise trails
LNORM=$2  # 0 for no length norm and 1 for length norm
NORMLOC=$3  # logps or logrs
NORMTYPE=$4  # scale or top or random
SEED=$5
RLType=$6  # ipo or sigmoid or kto_pair or dpop or mix or l1 or orpo or tdpo
ISLORA=0  # 1 for lora, 0 for full
M_TYPE=llama  # default llama, llama / qwen / baichuan2

DIRPRE=$7
DIRPOST=$8

raw_model_path=/row_model_path/
train_data_path=./data/roleMRC_train-rl_halfmix.jsonl
eval_data_path=./data/roleMRC_dev-rl.jsonl
deepspeed_config_path=./ds_config.json
model_output_path=/output_path/${RLType}_llama3_1-8B-${DIRPOST}-half_mix/
cache_path=./cache

if [ ${RLType} == "orpo" ]
then
    LR_TYPE=inverse_sqrt
    LR=$(awk "BEGIN {print 4e-6 * sqrt($WORLD_SIZE)}")
fi

if [ ${RLType} == "simpo" ]
then
    BETA=2.5
    EPOCH=1
fi

case ${raw_model_path} in 
    *"8B"*)
        PER_GPU_BATCH=4
        GRA_ACC=4
        ;;
    *"7B"*)
        PER_GPU_BATCH=4
        GRA_ACC=4
        ;;
    *"70b"*)
        PER_GPU_BATCH=1
        GRA_ACC=16
        ;;
esac

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
    --add_noise ${NOISE} \
    --len_norm ${LNORM} \
    --lnorm_loc ${NORMLOC} \
    --lnorm_type ${NORMTYPE} \
    --neft_alpha ${NEFTALPHA} \
    --dpop_lambda ${DPOP_LAMBADA} \
    --sft_lambda ${SFT_LAMBADA} \
    --l1_lambda ${L1_LAMBADA} \
    --simpo_lambda ${SIMPO_LAMBADA} \
    --tdpo_alpha ${TDPO_ALPHA} \
    --model_type ${M_TYPE} \
    --cache_dir ${cache_path} \
    --seed ${SEED}
