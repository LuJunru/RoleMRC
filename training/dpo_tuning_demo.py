import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import copy
import json

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed
)

from peft import (
    get_peft_model,
    LoraConfig
)
from peft.tuners.lora import LoraLayer
from trl import DPOTrainer

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."}
    )
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "The loss type."})
    if_lora: Optional[int] = field(default=0, metadata={"help": "Whether run lora or full training."})
    model_type: Optional[str] = field(default="llama", metadata={"help": "Model identification type."})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "beta value in DPO/IPO loss"})
    cache_dir: Optional[str] = field(default="", metadata={"help": "cache dir of datasets"})

@dataclass
class DataTrainingArguments:
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    train_data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "The input evaluation data file (a jsonlines)."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param

def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {
            k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k
        }
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return
    
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup seed
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # load config and tokenziers
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    # use truncation_side='left' to preserve linking between end of prompt and target labels
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left', trust_remote_code=True)
    label_ignore_id = -100
    # add pad token in tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def pack_llama3(example):
        system_text = "<|start_header_id|>system<|end_header_id|>\n\n" + example[0]["content"].strip() + "<|eot_id|>"
        if example[1]["role"] == "User":
            # Ensure that the assistant is the first speaker for the rest of the conversation
            system_text += "<|start_header_id|>user<|end_header_id|>\n\n"
            system_text += example[1]["content"].strip() + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            example = example[2:]
        else:
            system_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            example = example[1:]
        message_text = ""
        for m_i, message in enumerate(example):
            if m_i % 2 == 0:
                # The assistant speaks, and labels do not need masks
                message_text += message["content"].strip() + "<|eot_id|>"
            else:
                # User speaks, labels need masks
                message_text += "<|start_header_id|>user<|end_header_id|>\n\n" + message["content"].strip() + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return system_text + message_text
    
    def pack_Qwen2(example):
        system_text = "<|im_start|>system\n" + example[0]["content"].strip() + "<|im_end|>\n"
        if example[1]["role"] == "User":
            # Ensure that the assistant is the first speaker for the rest of the conversation
            system_text += "<|im_start|>user\n"
            system_text += example[1]["content"].strip() + "<|im_end|>\n" + "<|im_start|>assistant\n"
            example = example[2:]
        else:
            system_text += "<|im_start|>assistant\n"
            example = example[1:]
        message_text = ""
        for m_i, message in enumerate(example):
            if m_i % 2 == 0:
                # The assistant speaks, and labels do not need masks
                message_text += message["content"].strip() + "<|im_end|>\n"
            else:
                # User speaks, labels need masks
                message_text += "<|im_start|>user\n" + message["content"].strip() + "<|im_end|>\n" + "<|im_start|>assistant\n"
        return system_text + message_text
    
    def preprocess_function(examples, subset=None):
        prepared_inputs = {"prompt": [], "chosen": [], "rejected": []}
        for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            if "llama3" in model_args.model_name_or_path:
                p = pack_llama3(p)
            elif "Qwen2" in model_args.model_name_or_path:
                p = pack_Qwen2(p)
            prepared_inputs["prompt"].append(p)
            prepared_inputs["chosen"].append(c)
            prepared_inputs["rejected"].append(r)
        return prepared_inputs

    print("start data preprocess")
    data_files = {}
    data_files["train"] = data_args.train_data_path
    data_files["valid"] = data_args.eval_data_path
    if model_args.cache_dir != "":
        print("customized cache path: {}".format(model_args.cache_dir))
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    else:
        raw_datasets = load_dataset(
            "json",
            data_files=data_files
        )
    column_names = raw_datasets["train"].column_names
    prepared_dataset = {}
    prepared_dataset["train"] = raw_datasets["train"].map(
        lambda examples, indices: preprocess_function(examples, subset="train"),
        batched=True,
        batch_size=1024,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        with_indices=True,
        desc="Running tokenizer on training set"
    ).shuffle(seed=training_args.seed)
    prepared_dataset["valid"] = raw_datasets["valid"].map(
        lambda examples, indices: preprocess_function(examples, subset="valid"),
        batched=True,
        batch_size=1024,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        with_indices=True,
        desc="Running tokenizer on valid set"
    ).shuffle(seed=training_args.seed)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(prepared_dataset["train"]), data_args.max_train_samples)
        prepared_dataset["train"] = prepared_dataset["train"].select(range(max_train_samples))
    logger.info("load dataset finished, num of train: {}, num of valid: {}".format(len(prepared_dataset["train"]), len(prepared_dataset["valid"])))

    # initialize modules
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, use_flash_attention_2=True)
    if model_args.if_lora == 0:
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_flash_attention_2=True)
            
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    if len(tokenizer) > tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        if model_args.if_lora == 0:
            ref_model.resize_token_embeddings(len(tokenizer))

    # Setup Trainer
    training_args = training_args.to_dict()
    # training_args |= {'remove_unused_columns': False}
    training_args.update({'remove_unused_columns': False})
    training_args = TrainingArguments(**training_args)

    if model_args.model_type == "llama":
        target_modules = [
            "q_proj",
            "v_proj"
        ]
    elif model_args.model_type == "qwen":
        target_modules = [
            "c_attn"
        ]
    
    if model_args.if_lora != 0:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        ref_model = None

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=model_args.dpo_beta, # DPO temprature
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["valid"],
        tokenizer=tokenizer,
        args=training_args,
        max_length=data_args.model_max_length,
        max_prompt_length=int(data_args.model_max_length) * 3 // 4,
        loss_type=model_args.loss_type
    )

    # Training
    train_result = trainer.train()
    trainer.save_state()
    
    # save fp16 model under deepspeed zero2 or zero3
    c_stage = json.load(open(training_args.deepspeed, "r"))["zero_optimization"]["stage"]
    if c_stage in [2, 3]:
        if c_stage == 2:
            w_state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), "none")
        else:
            w_state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if trainer.is_world_process_zero():
            state_dict = {key: value.half().cpu() for key, value in w_state_dict.items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)
    else:
        trainer.save_model()

if __name__ == "__main__":
    main()
