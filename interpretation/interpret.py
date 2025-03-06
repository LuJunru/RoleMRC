import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import transformer_lens
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the device for PyTorch
torch.cuda.set_device(device)

def load_model(base_model_name, custom_model_path, device_map="balanced_low_0"):
    tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(custom_model_path, device_map="auto")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hooked_model = transformer_lens.HookedTransformer.from_pretrained(
        model_name=base_model_name,
        hf_model=hf_model,
        device_map="auto",
        dtype="bfloat16",
        attn_implementation="flash_attention_2"
    )
    # hooked_model.cuda()
    return tokenizer, hooked_model

def load_data(data_path):
    data = pd.read_json(data_path, lines=True)
    return data

def histogram_by_blocks(title, matrix, output_path):
    dimension_sum = []
    layers = []
    for idx, block in enumerate(matrix):
        # dimension_sum.append(sum(map(abs, block)))
        block = block / np.linalg.norm(block)
        dimension_sum.append(np.sum(np.abs(block, dtype=np.float64)))
        # dimension_sum.append(sum(block))
        layers.append(idx)

    plt.figure(figsize=(10, 6))
    plt.bar(layers, dimension_sum)
    plt.ylabel('Activation Sum')
    plt.xlabel('Layer')
    plt.title(f"{title} Activation Sum by Layer")
    plt.savefig(f"{output_path}/{title}_layer_active_bar.png", dpi=300)

def histogram_by_blocks_matrix(title, matrix):
    # Matrix is a numpy array
    dimension_sum = []
    layers = []
    for idx, block in enumerate(matrix):
        dimension_sum.append(sum(map(abs, block)))
        layers.append(idx)

    plt.figure(figsize=(10, 6))
    plt.bar(layers, dimension_sum)
    plt.ylabel('Activation Sum')
    plt.xlabel('Layer')
    plt.savefig(f"./figure/{title}_layer_active_bar.png", dpi=300)

def heat_map_by_blocks(title, matrix, output_path, save_npy=False):
    """
    Generate and save a heatmap for the given matrix.

    Parameters:
    - title (str): The title for the heatmap.
    - matrix (np.ndarray): 2D numpy array to visualize.
    - output_path (str): Directory to save the output.
    - save_npy (bool): Whether to save the matrix as an .npy file.
    """
    
    # Reduce precision and save as .npy file if needed
    if save_npy:
        np.save(f"{output_path}/{title}_activation_matrix.npy", matrix)
        print(f"Matrix saved: {output_path}/{title}_activation_matrix.npy")
        print(f"Matrix shape: {matrix.shape}")

    # Plot heatmap using seaborn
    plt.figure(figsize=(14, 5))
    sns.heatmap(matrix, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{title} Activation Visualization")
    plt.xlabel("Dimension")
    plt.ylabel("Layers")

    # Save the heatmap image
    plt.savefig(f"{output_path}/{title}_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()

    print(f"Heatmap saved: {output_path}/{title}_layer_active_heatmap.png")

def lin_heat_map(metrices, lengths, tasks, output_path):
    # Get matrix dims
    matrix = np.zeros((5, 32, 4096))
    prob = np.zeros((5, 32, 4096))
    entropy = np.zeros((32, 4096))

    matrix[0] = metrices[0]
    matrix[1] = metrices[1]
    matrix[2] = metrices[2]
    matrix[3] = metrices[3]
    matrix[4] = metrices[4]
    
    print_matrix = matrix
    # Display the data
    for j in range (0,32):
        for i in range (0,4096):
            prob[0][j][i] = matrix[0][j][i]/lengths[0]
            prob[1][j][i] = matrix[1][j][i]/lengths[1]
            prob[2][j][i] = matrix[2][j][i]/lengths[2]
            prob[3][j][i] = matrix[3][j][i]/lengths[3]
            prob[4][j][i] = matrix[4][j][i]/lengths[4]
            for k in range (0,5):
                entropy[j][i] = entropy[j][i] + prob[k][j][i] * (1000**(k+1))
        sorted_indices = np.argsort(entropy[j])[::-1]
        u = 0
        for s in sorted_indices:
            print_matrix[0][j][u] = matrix[0][j][s]
            print_matrix[1][j][u] = matrix[1][j][s]
            print_matrix[2][j][u] = matrix[2][j][s]
            print_matrix[3][j][u] = matrix[3][j][s]
            print_matrix[4][j][u] = matrix[4][j][s]
            u = u+1

    #plt.figure(figsize=(32, 4096))
    sns.heatmap(print_matrix[0], cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{tasks[0]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[0]}_ranked_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    sns.heatmap(print_matrix[1], cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{tasks[1]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[1]}_ranked_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    sns.heatmap(print_matrix[2], cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{tasks[2]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[2]}_ranked_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    sns.heatmap(print_matrix[3], cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{tasks[3]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[3]}_ranked_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    sns.heatmap(print_matrix[4], cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{tasks[4]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[4]}_ranked_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()

def hook_mlp_activations(hooked_model, inputs, mlp_hook_names):
    logits, activations = hooked_model.run_with_cache(
        inputs, 
        stop_at_layer=32, 
        names_filter=mlp_hook_names,
        return_cache_object=False
    )
    return activations

def filter_less_zero(matrix):
    # Filter less than 0
    matrix = np.where(matrix < 0, 0, matrix)
    return matrix

def normalize_by_row(matrix):
    # Normalize the activation by row
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix

def keep_top_k(matrix, k=5):
    # Keep the top k activations
    matrix = np.where(matrix < np.sort(matrix)[:, -k][:, None], 0, matrix)
    return matrix

def keep_top_percent(matrix, percent=0.05):
    # Keep the top k activations
    matrix = np.where(matrix < np.percentile(matrix, 100 - percent * 100, axis=1)[:, None], 0, matrix)
    return matrix

def count_non_zero_by_dim(matrix):
    # Count the number of non-zero activations by dimension
    count = np.count_nonzero(matrix, axis=0)
    return count

def count_by_top_percentage(matrix, percent=0.05):
    # Count the number of non-zero activations by dimension
    for each in matrix:
        count = np.count_nonzero(each > np.percentile(each, 100 - percent * 100))


def count_non_zero_by_block(matrix):
    # Count the number of non-zero activations by block
    count = np.count_nonzero(matrix, axis=1)
    return count

def count_non_zero_by_cell(matrix):
    # Count the number of non-zero activations by cell
    count = np.count_nonzero(matrix, axis=1)
    return count

mlp_hook_names = [f"blocks.{i}.hook_mlp_out" for i in range(32)] #LLaMA3.1 models 
mlp_hook_names = [f"blocks.{i}.hook_attn_out" for i in range(28)] #Qwen2.5 models 

task_to_dim = {
    "role_related_mrc_answer_with_narration": ["KNOWLEDGE_RANGE", "STYLE_COMPLIANCE"], # 2.1.1
    "role_related_mrc_answer_no_narration": ["KNOWLEDGE_RANGE"], # 2.1.2
    "role_unrelated_mrc_refused_with_narration": ["KNOWLEDGE_RANGE"], # 2.2.1
    "role_unrelated_mrc_refused_no_narration": ["KNOWLEDGE_RANGE", "STYLE_COMPLIANCE"], # 2.2.2
    "role_related_mrc_refused_with_narration": ["KNOWLEDGE_RANGE"], # 2.1.3
    "role_unrelated_mrc_answer_with_narration": ["KNOWLEDGE_RANGE"], # 2.2.3
    "role_related_mrc_refused_no_narration": ["STYLE_COMPLIANCE"], # 2.1.4
    "role_unrelated_mrc_answer_no_narration": ["STYLE_COMPLIANCE"], # 2.2.4
    "role_related_mrc_answer_with_narration-special-content": ["NESTED_INSTRUCTION"], # 3.1.1
    "role_related_mrc_answer_with_narration-special-format": ["NESTED_INSTRUCTION"], # 3.1.3
    "role_related_mrc_answer_no_narration-special-content": ["NESTED_INSTRUCTION"], # 3.1.2
    "role_related_mrc_answer_no_narration-special-format": ["NESTED_INSTRUCTION"], # 3.1.4
    "role_related_mrc_refused_with_narration-2ndrefused": ["MULTI_TURN_INSTRUCTION"], # 3.2.1
    "role_related_mrc_refused_no_narration-2ndrefused": ["MULTI_TURN_INSTRUCTION"], # 3.2.2
    "role_unrelated_mrc_refused_with_narration-2ndanswer": ["MULTI_TURN_INSTRUCTION"], # 3.2.3
    "role_unrelated_mrc_refused_no_narration-2ndanswer": ["MULTI_TURN_INSTRUCTION"], # 3.2.4
    "role_related_mrc_answer_with_narration-refused": ["INSTRUCTION_PRIORITY"], # 3.1.5
    "role_related_mrc_answer_no_narration-refused": ["INSTRUCTION_PRIORITY"] # 3.1.6
}

def main():
    # Load model and datasets
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    custom_model_path = "custom_model_path"
    data_path = "./data/RoleMRC/roleMRC_test.jsonl"
    output_path = "./qwen_sft"
    by_task = False

    tokenizer, hooked_model = load_model(base_model_name, custom_model_path)
    print(hooked_model.__dict__)
    data = pd.read_json(data_path, lines=True)
    # data = data[:100]

    task_lists = data["task"].unique()
    # Blocks Activation Matrix
    # create a dictionary, key is the task name, value is the activation matrix
    task_blocks_activation_matrix = {}

    # Intialize the task_blocks_activation_matrix
    if by_task:
        for task in task_lists:
            task_blocks_activation_matrix[task] = {}
            for name in mlp_hook_names:
                task_blocks_activation_matrix[task][name] = []
    else:
        for task in task_lists:
            dims = task_to_dim[task]
            for dim in dims:
                task_blocks_activation_matrix[dim] = {}
                for name in mlp_hook_names:
                    task_blocks_activation_matrix[dim][name] = []

    # Loop through all data
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Inference All Data"):
        question = row["question"]
        inputs = tokenizer.apply_chat_template(question, tokenize=False)
        task = row["task"]
        
        logits, activations = hooked_model.run_with_cache(
            inputs, 
            names_filter=mlp_hook_names,
            return_cache_object=False
        )
        # n is the number of tokens in the input
        # logits = n*128256

        for name, tensor in activations.items():
            # Access the specific tensor for this key and process it
            processed_tensor = tensor[-1][-1].to(torch.float16).cpu().numpy()
            if by_task:
                task_blocks_activation_matrix[task][name].append(processed_tensor)
            else:
                dims = task_to_dim[task]
                for dim in dims:
                    task_blocks_activation_matrix[dim][name].append(processed_tensor)
        del logits, activations
        hooked_model.reset_hooks()
    tasks = []
    lengths = []

    # Cast to float8 and save to npy file by dimension
    for task in task_blocks_activation_matrix.keys():
        print(f" Task: {task}, Number of samples: {len(task_blocks_activation_matrix[task][mlp_hook_names[0]])}")
        tasks.append(task)
        lengths.append(len(task_blocks_activation_matrix[task][mlp_hook_names[0]]))
        for name in mlp_hook_names:
            task_blocks_activation_matrix[task][name] = np.array(task_blocks_activation_matrix[task][name])
            task_blocks_activation_matrix[task][name] = task_blocks_activation_matrix[task][name].astype(np.float16)

        matrix = [task_blocks_activation_matrix[task][name] for name in mlp_hook_names]
        task_blocks_activation_matrix[task] = matrix
    
    # save to npy file
    np.save(f"{custom_model_path}/all_matrix.npy", task_blocks_activation_matrix)

if __name__ == "__main__":
    main()
