# RoleMRC: A Fine-Grained Composite Benchmark for Role-Playing and Instruction-Following.
Role-playing is important for Large Language Models (LLMs) to follow diverse instructions while maintaining role identity and the role's pre-defined ability limits. Existing role-playing datasets mostly contribute to controlling role style and knowledge boundaries, but overlook role-playing in instruction-following scenarios. We introduce a fine-grained role-playing and instruction-following composite benchmark, named RoleMRC, including: (1) Multi-turn dialogues between ideal roles and humans, including free chats or discussions upon given passages; (2) Role-playing machine reading comprehension, involving response, refusal, and attempts according to passage answerability and role ability; (3) More complex scenarios with nested, multi-turn and prioritized instructions. The final RoleMRC features a 10.2k role profile meta-pool, 37.9k well-synthesized role-playing instructions, and 1.4k testing samples. We develop a pipeline to quantitatively evaluate the fine-grained role-playing and instruction-following capabilities of several mainstream LLMs, as well as models that are fine-tuned on our data. Moreover, cross-evaluation on external role-playing datasets confirms that models fine-tuned on RoleMRC enhances instruction-following without compromising general role-playing and reasoning capabilities. We also probe the neural-level activation maps of different capabilities over post-tuned LLMs.

# Resources
- check our [paper](https://arxiv.org/abs/2502.11387), [data](https://huggingface.co/datasets/Junrulu/RoleMRC) and [local post-tuned models](https://huggingface.co/collections/Junrulu/rolemrc-67b2a4477a49eaea082ad33b). 
- check [training](training) , [evaluation](evaluation), and [interpretation](interpretation) codes, respectively.

# Straightforward Role-playing Example of RoleMRC
<img src="./rolemrc_example.png" width="988px"></img>

## Acknowledgment
This code is built upon the [TRL](https://github.com/huggingface/trl) repository.

## Citation
```
@article{LUandLI2025RoleMRC,
  title={RoleMRC: A Fine-Grained Composite Benchmark for Role-Playing and Instruction-Following},
  author={Lu, Junru and Li, Jiazheng and Shen, Guodong and Gui, Lin and An, Siyu and He, Yulan and Yin, Di and Sun, Xing},
  journal={arXiv preprint arXiv:2502.11387},
  year={2025}
}
```
