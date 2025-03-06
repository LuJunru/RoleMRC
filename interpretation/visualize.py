import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    if np.min(matrix) < 0:
        cmap = 'bwr'
    else:
        cmap = 'viridis'

    # Plot heatmap using seaborn
    plt.figure(figsize=(15, 5))
    heatmap = sns.heatmap(matrix, cmap=cmap, cbar=True,  xticklabels=False, yticklabels=True)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    if title == "INSTRUCTION_PRIORITY":
        sub_title = "Prioritized Instructions"
    elif title == "MULTI_TURN_INSTRUCTION":
        sub_title = "Multi-turn Instructions"
    elif title == "NESTED_INSTRUCTION":
        sub_title = "Nested Instructions"
    elif title == "STYLE_COMPLIANCE":
        sub_title = "Role Style"
    elif title == "KNOWLEDGE_RANGE":
        sub_title = "Knowledge Boundary"

    plt.title(f"{sub_title}")
    plt.xlabel("Dimension")
    plt.ylabel("Layers")

    # Save the heatmap image
    plt.savefig(f"{output_path}/{title}_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()

    print(f"Heatmap saved: {output_path}/{title}_layer_active_heatmap.png")

def lin_heat_map(all_matrix, metrices, tasks, output_path):
    # Get matrix dims
    matrix = np.zeros((5, 28, 3584))
    prob = np.zeros((5, 28, 3584))
    entropy = np.zeros((28, 3584))

    matrix[0] = metrices[0]
    matrix[1] = metrices[1]
    matrix[2] = metrices[2]
    matrix[3] = metrices[3]
    matrix[4] = metrices[4]
    
    print_matrix = matrix
    # Display the data
    for j in range (0,32):
        sorted_indices = np.argsort(all_matrix[j])[::-1]
        u = 0
        for s in sorted_indices:
            print_matrix[0][j][u] = matrix[0][j][s]
            print_matrix[1][j][u] = matrix[1][j][s]
            print_matrix[2][j][u] = matrix[2][j][s]
            print_matrix[3][j][u] = matrix[3][j][s]
            print_matrix[4][j][u] = matrix[4][j][s]
            u = u+1

    print("Sorted Indices:", sorted_indices)

    if np.min(print_matrix) < 0:
        cmap = 'bwr'
    else:
        cmap = 'viridis'

    #plt.figure(figsize=(32, 4096))
    plt.figure(figsize=(14, 5))
    sns.heatmap(print_matrix[0], cmap=cmap, cbar=True, xticklabels=False, yticklabels=True)
    plt.title(f"{tasks[0]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[0]}_lin_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    plt.figure(figsize=(14, 5))
    sns.heatmap(print_matrix[1], cmap=cmap, cbar=True,  xticklabels=False, yticklabels=True)
    plt.title(f"{tasks[1]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[1]}_lin_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    plt.figure(figsize=(14, 5))
    sns.heatmap(print_matrix[2], cmap=cmap, cbar=True,  xticklabels=False, yticklabels=True)
    plt.title(f"{tasks[2]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[2]}_lin_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    plt.figure(figsize=(14, 5))
    sns.heatmap(print_matrix[3], cmap=cmap, cbar=True,  xticklabels=False, yticklabels=True)
    plt.title(f"{tasks[3]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[3]}_lin_layer_active_heatmap.png", dpi=300)
    plt.tight_layout()
    plt.close()
    plt.figure(figsize=(14, 5))
    sns.heatmap(print_matrix[4], cmap=cmap, cbar=True,  xticklabels=False, yticklabels=True)
    plt.title(f"{tasks[4]} Activation Visualization")
    plt.savefig(f"{output_path}/{tasks[4]}_lin_layer_active_heatmap.png", dpi=300)
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

def filter_more_zero(matrix):
    # Filter less than 0
    matrix = np.where(matrix > 0, 0, matrix)
    return matrix

def normalize_by_row(matrix):
    # Normalize the activation by row
    matrix = matrix/matrix.sum(axis=1, keepdims=True)
    return matrix

def sum_by_row(matrix):
    # Sum the activation by row
    matrix = matrix.sum(axis=1)
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

def keep_top_percentage(matrix, percentage):
    """
    Keep the top `percentage`% of values in each row and set others to 0.
    
    :param matrix: NumPy 2D array.
    :param percentage: The percentage of top values to keep (0-100).
    :return: Modified matrix with only the top percentage values.
    """
    rows, cols = matrix.shape
    k = max(1, int(cols * (percentage / 100)))  # Number of elements to keep per row
    
    # Get the indices of the top-k values in each row
    sorted_indices = np.argsort(matrix, axis=1)[:, ::-1]  # Sort in descending order
    top_k_indices = sorted_indices[:, :k]  # Select top-k indices per row

    # Create a mask of the same shape as the matrix
    mask = np.zeros_like(matrix, dtype=bool)
    np.put_along_axis(mask, top_k_indices, True, axis=1)  # Set True for top-k indices

    # Apply the mask
    result = np.where(mask, matrix, 0)
    
    return result

def count_non_zero_by_dim(matrix):
    # Count the number of non-zero activations by dimension
    count = np.count_nonzero(matrix, axis=0)
    return count

def main():

    sft_matrix = np.load("sft-model-path/all_matrix.npy", allow_pickle=True)
    sft_matrix = sft_matrix.item()
    print(sft_matrix.keys())
    print(sft_matrix['KNOWLEDGE_RANGE'][0])
    print(sft_matrix['NESTED_INSTRUCTION'][0])
    # to numpy matrix
    sft_matrix = {task: np.array(sft_matrix[task]) for task in sft_matrix.keys()}

    dpo_matrix = np.load("dpo-model-path/all_matrix.npy", allow_pickle=True)
    dpo_matrix = dpo_matrix.item()
    dpo_matrix = {task: np.array(dpo_matrix[task]) for task in dpo_matrix.keys()}

    metrices = []
    tasks = []

    shape = (32, 4096)

    all_matrix = np.zeros(shape)
    all_length = []

    for task in sft_matrix.keys():
        print(task)

        matrix = sft_matrix[task]
        sft_count = []
        for each_layer in matrix:
            each_layer = keep_top_percentage(each_layer, 20)
            count = count_non_zero_by_dim(each_layer)
            sft_count.append(count)
        sft_count = np.array(sft_count)
        
        # visualize the matrix
        heat_map_by_blocks(task, sft_count, "./output/sft_visual/")

        matrix = dpo_matrix[task]
        dpo_count = []
        for each_layer in matrix:
            each_layer = keep_top_percentage(each_layer, 20)
            count = count_non_zero_by_dim(each_layer)
            dpo_count.append(count)
        dpo_count = np.array(dpo_count)
        
        # visualize the matrix
        heat_map_by_blocks(task, dpo_count, "./output/dpo_visual/")

        delta_count = dpo_count - sft_count
        abs_delta_count = np.abs(delta_count)

        # save non zero area
        non_zero_metrix = np.where(abs_delta_count > 0, 1, 0)

        # abs_delta_count = delta_count
        all_matrix = all_matrix + abs_delta_count
        all_length.append(matrix.shape[2])
        # normalize the matrix
        abs_delta_count = delta_count / matrix.shape[2]
        
        heat_map_by_blocks(task, abs_delta_count, "./output/discrepency/")
        metrices.append(abs_delta_count)
        tasks.append(task)

if __name__ == "__main__":
    main()
