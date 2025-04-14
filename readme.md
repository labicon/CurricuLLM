# CurricuLLM: Automatic Task Curricula Design for Learning Complex Robot Skills using Large Language Models

[[arXiv]](https://arxiv.org/abs/2409.18382)
[[Project Website]](https://iconlab.negarmehr.com/CurricuLLM/)

## Introduction

Curriculum learning is a training mechanism in reinforcement learning (RL) that facilitates the achievement of complex policies by progressively increasing the task difficulty during training. However, designing effective curricula for a specific task often requires extensive domain knowledge and human intervention, which limits its applicability across various domains. Our core idea is that large language models (LLMs), with their extensive training on diverse language data and ability to encapsulate world knowledge, present significant potential for efficiently breaking down tasks and decomposing skills across various robotics environments.

We propose CurricuLLM, which leverages the high-level planning and programming capabilities of LLMs for curriculum design, thereby enhancing the efficient learning of complex target tasks. CurricuLLM consists of: (Step 1) Generating sequence of subtasks that aid target task learning in natural language form, (Step 2) Translating natural language description of subtasks in executable task code, including the reward code and goal distribution code, and (Step 3) Evaluating trained policies based on trajectory rollout and subtask description. We evaluate CurricuLLM in various robotics simulation environments, ranging from manipulation, navigation, and locomotion, to show that CurricuLLM can aid learning complex robot control tasks.

## Installation

1. Create a conda environment
```
conda create -n CurricuLLM python=3.10
conda activate CurricuLLM
```

2. Install CurricuLLM

Before you run CurricuLLM, please install dependencies and edited custom environments.  
```
git clone https://github.com/labicon/CurricuLLM.git
cd CurricuLLM
pip install -r requirements.txt
pip install -e environments
```

3. OpenAI API

CurricuLLM currently uses OpenAI API for LLM interaction. Please add your personal API key in `./gpt/key.yaml` as
```
OPENAI_API_KEY: your API key
```

## Getting Started

You can run CurricuLLM using `main.py`
```
python main.py --task={Task name} --exp={Experiment name} --logdir={Log directory} --seed={random seed}
```

* `task` is a task to learn. You can find options in `configs`
* `exp` is experiment options. Currently, there are 4 experiment options.
    * curriculum: CurricuLLM experiments
    * her: Hindsight Experience Replay baseline
    * sac: Soft Actor Critic baseline
    * zeroshot: LLM-zeroshot baseline
* `logdir` is directory that you want to store the results
* `seed` is random seed for your experiments

## Acknowledgement
* Our RL training is based on stable-baselines3
* Our environments are from gymnasium-robotics
