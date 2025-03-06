import sys
import os
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from openai import AsyncOpenAI
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./output/gpt4_judge.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Evalution Dimention 1: Role Knowledge Range Compliance
KNOWLEDGE_RANGE = """
# An LLM is playing a specific character and answer a question about the given passages:
{conversations}
# The LLM generates the following response of the question:
\"{response}\"
# Ideally, the LLM should generate a response according to the character's **knowledge range**.
## 1. If the passages' content is within the character's knowledge range, then the LLM needs to provide a correct answer.
## 2. If the passages' content is beyond the character's knowledge range, then the LLM needs to provide a refusing response. Except the LLM is required to give an answer in any case.
# Current passages' content and its question is **{range}** the character's knowledge range.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response incorrectly answers the question that is beyond the character's knowledge range, or refuses to answer a question that is within the character's knowledge range.
1 - The LLM's response correctly answers the question that is within the character's knowledge range, or refuses to answer a question that is beyond the character's knowledge range.
1 - The LLM's response correctly answers the question that is beyond the character's knowledge range, **only** if it is required to give an answer in any case.
Score:
"""

# Evalution Dimention 2: Role Style Compliance
STYLE_COMPLIANCE = """
# An LLM is playing a specific character and answer a question about the given passages:
{conversations}
# The LLM generates the following response of the question:
\"{response}\"
# Ideally, the LLM should generate a response that is consistent with the **character's speech style**.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response lacks the main characteristics of the character's speaking style.
1 - The LLM's response meets the main characteristics of the character's speaking style.
Score:
"""

# Evalution Dimention 3: Nested Instruction Compliance
NESTED_INSTRUCTION = """
# An LLM is playing a specific character and answer a question about the given passages. Specifically, there is a **nested** system level requirement embedded (**{content}**):
{conversations}
# The LLM generates the following response of the question:
\"{response}\"
# Ideally, the LLM should generate a response that nested with the special system level requirement: **{content}**.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response does not follow the nested instruction (**{content}**). 
1 - The LLM's response **follows** the nested instruction (**{content}**).
Score:
"""

# Evalution Dimention 4: Multi-turn Instruction Compliance
MULTI_TURN_INSTRUCTION = """
# An LLM is playing a specific character and answer a question about the given passages. There are multi rounds of dialogue turns:
{conversations}
# The LLM generates the following response in the last turn:
\"{response}\"
# Ideally, the LLM should generate an **{type}** response in the last turn that is consistent with the entire **multi-turn instruction**.
# Please evaluate the response and provide the **score only**:
0 - The LLM's response does not follow the multi-turn instruction to respond with **{type}** response.
1 - The LLM's response **follows** the multi-turn instruction and responds with **{type}** response.
Score:
"""

# Evalution Dimention 5: Instruction Priority Compliance
INSTRUCTION_PRIORITY = """
# An LLM is playing a specific character and answer a question about the given passages. Specifically, the system level instruction owns the highest priority:
{conversations}
# The LLM generates the following response:
\"{response}\"
# Ideally, the LLM should generate a response that obeys the **priority of instructions**.
## 1. The system's instruction own the highest priority.
## 2. The user's instruction own the second highest priority.
# Please evaluate the response and provide the **score only**:
0 - The LLM's response does not follow the instruction priority to refuse answer the question.
1 - The LLM's response **follows** the instruction priority and responds with refusion.
Score:
"""

evaluation_config = {
    "role_related_mrc_answer_with_narration": [{"template": KNOWLEDGE_RANGE, "range": "within"}, {"template": STYLE_COMPLIANCE}], # 2.1.1
    "role_related_mrc_answer_no_narration": [{"template": KNOWLEDGE_RANGE, "range": "within"}], # 2.1.2
    "role_unrelated_mrc_refused_with_narration": [{"template": KNOWLEDGE_RANGE, "range": "outside"}], # 2.2.1
    "role_unrelated_mrc_refused_no_narration": [{"template": KNOWLEDGE_RANGE, "range": "outside"}, {"template": STYLE_COMPLIANCE}], # 2.2.2
    "role_related_mrc_refused_with_narration": [{"template": KNOWLEDGE_RANGE, "range": "within"}], # 2.1.3
    "role_unrelated_mrc_answer_with_narration": [{"template": KNOWLEDGE_RANGE, "range": "outside"}], # 2.2.3
    "role_related_mrc_refused_no_narration": [{"template": STYLE_COMPLIANCE}], # 2.1.4
    "role_unrelated_mrc_answer_no_narration": [{"template": STYLE_COMPLIANCE}], # 2.2.4
    "role_related_mrc_answer_with_narration-special-content": [{"template": NESTED_INSTRUCTION}], # 3.1.1
    "role_related_mrc_answer_with_narration-special-format": [{"template": NESTED_INSTRUCTION}], # 3.1.3
    "role_related_mrc_answer_no_narration-special-content": [{"template": NESTED_INSTRUCTION}], # 3.1.2
    "role_related_mrc_answer_no_narration-special-format": [{"template": NESTED_INSTRUCTION}], # 3.1.4
    "role_related_mrc_refused_with_narration-2ndrefused": [{"template": MULTI_TURN_INSTRUCTION, "type": "unanswerable"}], # 3.2.1
    "role_related_mrc_refused_no_narration-2ndrefused": [{"template": MULTI_TURN_INSTRUCTION, "type": "unanswerable"}], # 3.2.2
    "role_unrelated_mrc_refused_with_narration-2ndanswer": [{"template": MULTI_TURN_INSTRUCTION, "type": "answerable"}], # 3.2.3
    "role_unrelated_mrc_refused_no_narration-2ndanswer": [{"template": MULTI_TURN_INSTRUCTION, "type": "answerable"}], # 3.2.4
    "role_related_mrc_answer_with_narration-refused": [{"template": INSTRUCTION_PRIORITY}], # 3.1.5
    "role_related_mrc_answer_no_narration-refused": [{"template": INSTRUCTION_PRIORITY}] # 3.1.6
}

class OALLM:
    def __init__(self, model, temperature=0.7, max_length=256):
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.client = AsyncOpenAI(api_key="please-input-your-openai-api-key")

    async def generate(self, batch_messages):
        semaphore = asyncio.Semaphore(5)
        progress_bar = tqdm(total=len(batch_messages))
        tasks = [
            asyncio.create_task(
                self.query(messages["prompt"], semaphore, progress_bar)
            ) for messages in batch_messages
        ]
        results = await asyncio.gather(*tasks)
        progress_bar.close()
        return results
    
    async def query(self, messages, semaphore, progress_bar):
        async with semaphore:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                n=1,
                max_tokens=self.max_length
            )
            progress_bar.update(1)
            return completion.choices[0].message.content

def build_conversation(conversation):
    message = ""
    for idx, turn in enumerate(conversation):
        if turn["role"].lower() == "system":
            message += f"System Instruction: \"{turn['content']}\"\n"
        if turn["role"].lower() == "user":
            message += f"User Query: \"{turn['content']}\"\n"
        if turn["role"].lower() == "assistant":
            message += f"LLM Response: \"{turn['content']}\"\n"
    return message

# Main evaluation loop
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4-turbo-2024-04-09", help="OpenAI model as the judge.")
    parser.add_argument("--test_path", type=str, default="", help="Tested data path to evaluate.")
    parser.add_argument("--tempreture", type=float, default=0.7, help="Tempreture for the model.")
    parser.add_argument("--max_length", type=int, default=4, help="Max generation length.")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    args = parser.parse_args() 

    if os.path.exists(args.test_path):
        test_data = pd.read_json(args.test_path, lines=True)
        for each in ["task", "question", "response"]:
            if each not in test_data.columns:
                raise ValueError(f"Column {each} not found in the test data.")
    else:
        raise ValueError(f"Test data not found at {args['test_path']}.")

    if args.debug:
        test_data = test_data[:10]

    # Build tests
    all_queries = []
    for idx, row in test_data.iterrows():
        task = row["task"]
        original_conversation = row["question"]
        response = row["response"]

        conversation = build_conversation(original_conversation)

        tasks = evaluation_config[task]
        for task in tasks:
            if task["template"] == KNOWLEDGE_RANGE:
                task_name = "knowledge_range"
                prompt = task["template"].format(conversations=conversation, response=response, range=task.get("range", ""))
            elif task["template"] == NESTED_INSTRUCTION:
                task_name = "nested_instruction"
                content = original_conversation[0]['content'].split(". ")[1].replace("You love to","").replace("You will","").replace("You must", "").replace("You prefer to","").replace("You would like to","").replace("You are used to", "").replace("You should", "").replace("You are in the habit of","")
                if content[-1] == ".":
                    content = content[:-1]
                prompt = task["template"].format(conversations=conversation, response=response, content=content)
            elif task["template"] == MULTI_TURN_INSTRUCTION:
                task_name = "multi_turn_instruction"
                prompt = task["template"].format(conversations=conversation, response=response, type=task.get("type", ""))
            else:
                prompt = task["template"].format(conversations=conversation, response=response)
                if task["template"] == STYLE_COMPLIANCE:
                    task_name = "style_compliance"
                elif task["template"] == INSTRUCTION_PRIORITY:
                    task_name = "instruction_priority"
                    all_queries.append({"task": task_name, "prompt": [{"role": "user", "content": prompt}]})
                else:
                    raise ValueError(f"Unknown task template {task['template']}.")
            all_queries.append({"task": task_name, "prompt": [{"role": "user", "content": prompt}]})
    
    print(f"Generated {len(all_queries)} queries for evaluation. Test set size: {len(test_data)}.")

    # Initialize the LLM
    llm = OALLM(args.model, temperature=args.tempreture, max_length=args.max_length)
    responses = await llm.generate(all_queries)

    # Evaluate the responses
    by_aspects = {
        "knowledge_range":[],
        "style_compliance":[],
        "nested_instruction":[],
        "multi_turn_instruction":[],
        "instruction_priority":[]
    }
    bad_responses = {
        "knowledge_range": 0,
        "style_compliance": 0,
        "nested_instruction": 0,
        "multi_turn_instruction": 0,
        "instruction_priority": 0
    }
    for idx, response in enumerate(responses):
        task_name = all_queries[idx]["task"]
        try:
            if "Score:" in response:
                response = response.replace("Score:", "")
            response_score = int(response)
        except:
            response_score = 0
            bad_responses[task_name] += 1
        by_aspects[task_name].append(response_score)

    logger.info(f"Evaluated {args.test_path}, Model: {args.model}, Tempreture: {args.tempreture}, Max Length: {args.max_length}.")
    filename = args.test_path.split("/")[-1]
    result_path  = f"./logs/llm_as_judge.jsonl"
    
    # read json and append new results
    with open(result_path, "r") as f:
        summary = pd.read_json(f, lines=True)
    # idx = 1
    # while filename in summary["filename"].values:
    #     filename = filename.replace(".jsonl", "")
    #     filename += f"_{idx}.jsonl"
    #     idx += 1
    
    for aspect in by_aspects.keys():
        length = len(by_aspects[aspect])
        if length == 0:
            by_aspects[aspect] = 0
            print(f"Aspect {aspect} score: {by_aspects[aspect]} (no data)")
            logger.info(f"- aspect {aspect} score: {by_aspects[aspect]} (no data)")
        else:
            by_aspects[aspect] = sum(by_aspects[aspect]) / length
            print(f"Aspect {aspect} score: {by_aspects[aspect]}, bad responses: {bad_responses[aspect]}/{length}")
            logger.info(f"- aspect {aspect} score: {by_aspects[aspect]}, bad responses: {bad_responses[aspect]}/{length}")
    
    new_row = pd.DataFrame([{
        "filename": filename,
        "knowledge_range": by_aspects["knowledge_range"],
        "style_compliance": by_aspects["style_compliance"],
        "nested_instruction": by_aspects["nested_instruction"],
        "multi_turn_instruction": by_aspects["multi_turn_instruction"],
        "instruction_priority": by_aspects["instruction_priority"]
    }])

    summary = pd.concat([summary, new_row], ignore_index=True)

    with open(result_path, "w") as f:
        summary.to_json(f, orient="records", lines=True)
    logger.info(f"Results saved to {result_path}.")

    # Save the responses
    all_queries = pd.DataFrame(all_queries)
    all_queries["response"] = responses
    all_queries.to_json(f"./llm_judge/{filename}", orient="records", lines=True)
    logger.info(f"Responses saved to evaluation/llm_judge/{filename}.")

if __name__ == "__main__":
    asyncio.run(main())