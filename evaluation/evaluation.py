import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# If model stucked at loading, try to disable NCCL_P2P
# os.environ["NCCL_P2P_DISABLE"] = "1"
import argparse
import json
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, EngineArgs
from dataclasses import dataclass, asdict
from datetime import datetime
import evaluate

from openai import OpenAI
import asyncio
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

import sys

# Increase the recursion limit
sys.setrecursionlimit(3000)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

METRICS_SUMMARY_COLUMNS = {
    "ROUGE": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    "BLEU": ["bleu"],
    "METEOR": ["meteor"],
    "BERTScore": ["f1"]
}

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]

AVALIABLE_TASKS = ["RoleMRC", "RoleBenchInstEng", "RoleBenchRoleEng"]

@dataclass
class EvaluationArgs:
    test_file_path: str
    input_column: str
    reference_column: str
    metrics: list

rolemrc_configs = EvaluationArgs(
    test_file_path="./data/RoleMRC/roleMRC_test.jsonl",
    input_column="question",
    reference_column="reference",
    metrics=["BLEU", "ROUGE", "METEOR", "BERTScore"]
)

rolebench_inst_eng_configs = EvaluationArgs(
    test_file_path="./data/RoleBench/instruction_generalization_test.jsonl",
    input_column="question",
    reference_column="reference",
    metrics=["ROUGE"]
)

rolebench_role_eng_configs = EvaluationArgs(
    test_file_path="./data/RoleBench/role_generalization_test.jsonl",
    input_column="question",
    reference_column="reference",
    metrics=["ROUGE"]
)

class OALLM:
    def __init__(self, model):
        self.model = model
        api_key_path = "./api_key.txt"
        if not os.path.exists(api_key_path):
            raise ValueError(f"API key file not found at {api_key_path}.")
        else:
            self.client = OpenAI(api_key_path=api_key_path)

    async def generate(self, batch_messages, sampling_params):
        temp, max_length = sampling_params.temperature, sampling_params.max_tokens
        semaphore = asyncio.Semaphore(20)
        progress_bar = tqdm(total=len(batch_messages))
        tasks = [
            self.query(
                messages=messages, 
                temp=temp,
                max_length=max_length,
                semaphore=semaphore,
                progress_bar=progress_bar)
            for messages in batch_messages
        ]
        results = await asyncio.gather(*tasks)
        progress_bar.close()
        return results
    
    async def query(self, messages, temp=0.0, max_length=256, semaphore=None, progress_bar=None):
        async with semaphore:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=temp,
                messages=messages,
                n=1,
                max_tokens=max_length
            )
            progress_bar.update(1)
            return completion.choices[0].message.content

rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
async def evaluate_rouge(reference, response, row_idx):
    try:
        rouge_scores = rouge_score.score(response, reference)
        return {'rouge1': rouge_scores['rouge1'].fmeasure, 'rouge2': rouge_scores['rouge2'].fmeasure, 'rougeL': rouge_scores['rougeL'].fmeasure, 'rougeLsum': rouge_scores['rougeLsum'].fmeasure}
    except Exception as e:
        logger.error(f"Error in evaluate_rouge: {e} at row {row_idx}.")
        logger.error(f"Response: {response}; Reference: {reference}.")
        return np.nan

# BLEU score
bleu = evaluate.load("bleu")
async def evaluate_bleu(reference, response, row_idx):
    try:
        return bleu.compute(predictions=[response], references=[[reference]])
    except Exception as e:
        logger.error(f"Error in evaluate_bleu: {e} at row {row_idx}.")
        logger.error(f"Response: {response}; Reference: {reference}.")
        return np.nan

# METEOR score
meteor_score = evaluate.load('meteor')
async def evaluate_meteor(reference, response, row_idx):
    try:
        return meteor_score.compute(predictions=[response], references=[reference])
    except Exception as e:
        logger.error(f"Error in evaluate_meteor: {e} at row {row_idx}.")
        logger.error(f"Response: {response}; Reference: {reference}.")
        return np.nan

# BERTScore
bertscore = evaluate.load("bertscore")
async def evaluate_bert(reference, response, row_idx):
    try:
        bert_scores = bertscore.compute(predictions=[response], references=[reference], lang="en")
        return {'precision': bert_scores['precision'][0], 'recall': bert_scores['recall'][0], 'f1': bert_scores['f1'][0]}
    except Exception as e:
        logger.error(f"Error in evaluate_bert: {e} at row {row_idx}.")
        logger.error(f"Response: {response}; Reference: {reference}.")
        return np.nan

async def evaluate_metrics(references, responses, metrics):
    results = []
    row_idx = 0
    async for reference, response in async_tqdm(zip(references, responses), total=len(references), desc="Evaluating Metrics"):
        result = {}
        tasks = []
        metric_map = {}

        # Create tasks and map them to their metrics
        if "ROUGE" in metrics:
            task = evaluate_rouge(reference, response, row_idx)
            tasks.append(task)
            metric_map[task] = 'ROUGE'
        if "BLEU" in metrics:
            task = evaluate_bleu(reference, response, row_idx)
            tasks.append(task)
            metric_map[task] = 'BLEU'
        if "METEOR" in metrics:
            task = evaluate_meteor(reference, response, row_idx)
            tasks.append(task)
            metric_map[task] = 'METEOR'
        if "BERTScore" in metrics:
            task = evaluate_bert(reference, response, row_idx)
            tasks.append(task)
            metric_map[task] = 'BERTScore'

        # Gather all results concurrently
        results_concurrent = await asyncio.gather(*tasks, return_exceptions=True)

        # Record results, handle exceptions
        for task, result_val in zip(tasks, results_concurrent):
            metric = metric_map[task]
            if isinstance(result_val, Exception):
                logger.error(f"Error processing {metric} at row {row_idx} for entry: {result_val}, set to 0.0")
                result[metric] = 0.0
            else:
                result[metric] = result_val

        row_idx += 1
        results.append(result)
    return results

def load_dataset(data_path):
    messages = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            messages.append(json.loads(line.strip()))

    # replace System, User, and Assistant to system, user, and assistant
    for each_row in messages:
        for each_turn in each_row['question']:
            if each_turn['role'] == 'System':
                each_turn['role'] = 'system'
            elif each_turn['role'] == 'User':
                each_turn['role'] = 'user'
            elif each_turn['role'] == 'Assistant':
                each_turn['role'] = 'assistant'
    
    return messages

# RolePlay Evaluation
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="RoleMRC", help="Test to run.")
    parser.add_argument("--model", type=str, default="1", help="Model to evaluate.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor_parallel_size for vLLM.")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    # parser.add_argument("--pre_tested", type=str, default="", help="Whether the model has been pre-tested.")
    args = parser.parse_args()

    if args.debug:
        print("Debug mode enabled.")
    
    # Pre Configurations for the test task
    test_task = args.tasks
    if "," in test_task:
        test_task = test_task.split(",")
    else:
        test_task = [test_task]

    sampling_params = SamplingParams(
        n=1, 
        temperature=0.0, 
        max_tokens=256
    )

    # Load the model and preprocess the data
    model_name = args.model
    
    if model_name in OPENAI_MODELS:
        llm = OALLM(model_name)
    else:
        # vLLM Configs
        vllm_configs = EngineArgs(
            model=model_name,
            gpu_memory_utilization=0.90, 
            tensor_parallel_size=args.tensor_parallel_size, 
            max_num_batched_tokens=4096, 
            max_model_len=4096, 
            enforce_eager=False, 
            trust_remote_code=True
        )

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)    
        # Initialize the model
        llm = LLM(**asdict(vllm_configs))

    # Start logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"./logs/eval.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # save all configs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    with open(f"./logs/configs/{model_name.split('/')[-1]}_{timestamp}_config.json", 'w') as file:
        if model_name in OPENAI_MODELS:
            all_configs = {
                "parser_args": vars(args),
                "sampling_params": vars(sampling_params)
            }
        else:
            all_configs = {
                "parser_args": vars(args),
                "vllm_configs": vars(vllm_configs),
                "sampling_params": vars(sampling_params)
            }
        file.write(json.dumps(all_configs))

    for test_task in test_task:
        if test_task == "RoleMRC":
            test_config = rolemrc_configs
        elif test_task == "RoleBenchInstEng":
            test_config = rolebench_inst_eng_configs
        elif test_task == "RoleBenchRoleEng":
            test_config = rolebench_role_eng_configs
        else:
            raise ValueError(f"Test task {test_task} not recognized, avaliable tasks are: {AVALIABLE_TASKS} ")

        # Load and preprocess the dataset
        data = load_dataset(test_config.test_file_path)
        
        if args.debug:
            data = data[:10]

        logger.info(f"Evaluating {model_name} with {test_task}: total {len(data)} instances.")

        if model_name in OPENAI_MODELS:
            questions = [each[test_config.input_column] for each in data]
            # generate outputs
            outputs = await llm.generate(questions, sampling_params)
            logger.info(f"Generation finished with {len(outputs)} instances.")
            responses = outputs
        else:
            questions = [tokenizer.apply_chat_template(each[test_config.input_column], tokenize=False, add_generation_prompt=True) for each in data]
            # generate outputs
            outputs = llm.generate(questions, sampling_params)
            logger.info(f"Generation finished with {len(outputs)} instances.")
            responses = [output.outputs[0].text for output in outputs]

        # Add responses to the data
        for row, response in zip(data, responses):
            row['response'] = response
        
        # Start calculating the metrics with asycn
        reference_column = test_config.reference_column
        references = [each[reference_column] for each in data]

        if args.debug:
            print('References', references[:10])
            print('Responses', responses[:10])
            print(rouge_score.score(responses[0], references[0]))

        # Save the generated responses
        saving_path = f"./output/eval_{model_name.split('/')[-1]}_{test_task}.jsonl"
        with open(saving_path, 'w') as file:
            for row in data:
                file.write(json.dumps(row) + '\n')
        logger.info(f"Generated responses saved to {saving_path}")

        metrics = test_config.metrics

        evaluation_results = await evaluate_metrics(references, responses, metrics)

        # Save the evaluation results
        saving_path = f"./output/eval_{model_name.split('/')[-1]}_{test_task}_results.jsonl"
        with open(saving_path, 'w') as file:
            for eval_row, row in zip(evaluation_results, data):
                row.update(eval_row)
                file.write(json.dumps(row) + '\n')
        logger.info(f"Evaluation results saved to {saving_path}")

        # Summarize results
        summary_stats = {}
        for metric in metrics:
            sub_metrics = METRICS_SUMMARY_COLUMNS[metric] 
            summary_stats[metric] = {sub_metric: {'mean': 0, 'std_dev': 0} for sub_metric in sub_metrics}

        # Process each metric and sub-metric
        for metric in metrics:
            sub_metrics = METRICS_SUMMARY_COLUMNS[metric]
            for sub_metric in sub_metrics:
                # Extract relevant values for each sub-metric
                values = []
                for result in evaluation_results:
                    if isinstance(result[metric], dict) and sub_metric in result[metric]:
                        value = result[metric][sub_metric]
                        if isinstance(value, (int, float)):
                            values.append(value)

                # Calculate mean and standard deviation if values are present
                if values:
                    mean_value = np.mean(values)
                    std_dev_value = np.std(values, ddof=0)  # Population standard deviation
                    summary_stats[metric][sub_metric]['mean'] = mean_value
                    summary_stats[metric][sub_metric]['std_dev'] = std_dev_value
                else:
                    summary_stats[metric][sub_metric]['mean'] = None
                    summary_stats[metric][sub_metric]['std_dev'] = None

        # Logging summary statistics
        logger.info(f"Summary of results: Model: {model_name}, Task: {test_task}")
        for metric, sub_metrics in summary_stats.items():
            for sub_metric, stats in sub_metrics.items():
                if stats['mean'] is not None:
                    logger.info(f"{metric} {sub_metric} - Mean: {stats['mean']:.4f}, Standard   Deviation: {stats['std_dev']:.4f}")
                    print(f"{metric} {sub_metric} - Mean: {stats['mean']:.4f}, Standard     Deviation: {stats['std_dev']:.4f}")
                else:
                    logger.info(f"{metric} {sub_metric} - Mean: N/A, Standard Deviation: N/A")
                    print(f"{metric} {sub_metric} - Mean: N/A, Standard Deviation: N/A")

        # Write to file
        with open(f"./logs/results/{model_name.split('/')[-1]}_{test_task}_results.json", 'w') as file:
            file.write(json.dumps(summary_stats, default=str))  # Handle None types
    

# Example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --tasks RoleMRC/RoleBenchInstEng/RoleBenchRoleEng --model meta-llama/Llama-3.1-8B-Instruct --tensor_parallel_size 4
if __name__ == "__main__":
    asyncio.run(main())