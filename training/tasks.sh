#!/bin/bash

# SFT
bash sft_llama3.sh 4200 roleMRC_train-sft_mix.jsonl RoleMRC 8 2 ds_config.json 8B None > llama31_8b_mix.log
# DPO
bash dpo_llama3.sh 0 0 logps random 42 sigmoid raw dpo > llama31_8b_dpo_halfmix.log
