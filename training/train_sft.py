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
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

from peft import (
    get_peft_model,
    LoraConfig
)

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    if_lora: Optional[int] = field(default=0, metadata={"help": "Whether run lora or full training."})
    cache_dir: Optional[str] = field(default="/root/cache", metadata={"help": "cache dir of datasets"})

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
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
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

    # load dataset
    logger.info("start data preprocess")
    label_ignore_id = -100
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    # add pad token in tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples, indices=None):
        model_inputs = {"input_ids": [], "labels": []}
        for e_i, example in enumerate(examples["conversations"]):
            input_ids = []
            label_ids = []
            if "llama" in model_args.model_name_or_path.lower():
                system_text = "<|start_header_id|>system<|end_header_id|>\n\n" + example[0]["content"].strip() + "<|eot_id|>"
            elif "Qwen2" in model_args.model_name_or_path:
                system_text = "<|im_start|>system\n" + example[0]["content"].strip() + "<|im_end|>\n"
            if example[1]["role"] == "User":
                # Ensure that the assistant is the first speaker for the rest of the conversation
                if "llama" in model_args.model_name_or_path.lower():
                    system_text += "<|start_header_id|>user<|end_header_id|>\n\n"
                    system_text += example[1]["content"].strip() + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                elif "Qwen2" in model_args.model_name_or_path:
                    system_text += "<|im_start|>user\n"
                    system_text += example[1]["content"].strip() + "<|im_end|>\n" + "<|im_start|>assistant\n"
                example = example[2:]
            else:
                if "llama" in model_args.model_name_or_path.lower():
                    system_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                elif "Qwen2" in model_args.model_name_or_path:
                    system_text += "<|im_start|>assistant\n"
                example = example[1:]
            system_ids = tokenizer.encode(system_text, add_special_tokens=False)
            if bos_token is not None:
                system_ids = [tokenizer.bos_token_id] + system_ids
            input_ids.extend(system_ids)
            label_ids.extend([label_ignore_id] * len(system_ids))
            for m_i, message in enumerate(example):
                if m_i % 2 == 0:
                    # The assistant speaks, and labels do not need masks
                    if "llama" in model_args.model_name_or_path.lower():
                        message_text = message["content"].strip() + "<|eot_id|>"
                    elif "Qwen2" in model_args.model_name_or_path:
                        message_text = message["content"].strip() + "<|im_end|>\n"
                    message_ids = tokenizer.encode(message_text, add_special_tokens=False)
                    input_ids.extend(message_ids)
                    label_ids.extend(message_ids)
                else:
                    # User speaks, labels need masks
                    if "llama" in model_args.model_name_or_path.lower():
                        message_text = "<|start_header_id|>user<|end_header_id|>\n\n" + message["content"].strip() + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    elif "Qwen2" in model_args.model_name_or_path:
                        message_text = "<|im_start|>user\n" + message["content"].strip() + "<|im_end|>\n" + "<|im_start|>assistant\n"
                    message_ids = tokenizer.encode(message_text, add_special_tokens=False)
                    input_ids.extend(message_ids)
                    label_ids.extend([label_ignore_id] * len(message_ids))
            input_ids.append(tokenizer.eos_token_id)
            label_ids.append(tokenizer.eos_token_id)
            if len(input_ids) >= data_args.model_max_length:
                model_inputs["input_ids"].append(input_ids[:data_args.model_max_length])
                model_inputs["labels"].append(label_ids[:data_args.model_max_length])
            else:
                padding_len = data_args.model_max_length - len(input_ids)
                model_inputs["input_ids"].append(input_ids + [tokenizer.pad_token_id] * padding_len)
                model_inputs["labels"].append(label_ids + [label_ignore_id] * padding_len)
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["labels"] = torch.tensor(model_inputs["labels"])
        model_inputs["attention_mask"] = model_inputs["input_ids"].ne(tokenizer.pad_token_id)
        return model_inputs

    data_files = {}
    data_files["train"] = data_args.train_data_path
    # data_files["valid"] = data_args.eval_data_path
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir
    )
    column_names = raw_datasets["train"].column_names

    prepared_dataset = {}
    prepared_dataset["train"] = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        batch_size=1024,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        with_indices=True,
        desc="Running tokenizer on training set"
    ).shuffle(seed=training_args.seed)
    prepared_dataset["train"] = prepared_dataset["train"].shuffle(seed=training_args.seed)
    prepared_dataset["train"].set_format(type="pt")
    prepared_dataset["train"] = prepared_dataset["train"].filter(lambda example: (example['labels'] != label_ignore_id).any())

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(prepared_dataset["train"]), data_args.max_train_samples)
        prepared_dataset["train"] = prepared_dataset["train"].select(range(max_train_samples))
    # logger.info("load dataset finished, num of train: {}, num of valid: {}".format(len(prepared_dataset["train"]), len(prepared_dataset["valid"])))
    logger.info("load dataset finished, num of train: {}, num of valid: 0".format(len(prepared_dataset["train"])))

    # initialize modules
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    if len(tokenizer) > tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # lora
    if model_args.if_lora != 0:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        # eval_dataset=prepared_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    # Training
    if "steps" in training_args.resume_from_checkpoint or "checkpoint" in training_args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
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
