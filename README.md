# NeurIPS-llm-efficiency-challenge
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Model Weight License](https://img.shields.io/badge/Model%20Weights%20License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

[Our model](https://huggingface.co/upaya07/model_02) won 🏆 first prize 🏆 in RTX 4090 track in [NeurIPS Large Language Model Efficiency Challenge: 1 LLM + 1GPU + 1Day](https://llm-efficiency-challenge.github.io/) competition. We used [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) as a base model and used QLoRA to fine-tune it for 24 hours on a single RTX 4090 GPU.
<br>

| Model Name | Checkpoint | Dataset  | License|
| ----- |------| ---- | ----- |
| Birbal-7B-V1 | 🤗 <a href="https://huggingface.co/upaya07/Birbal-7B-V1" target="_blank">Birbal-7B-V1</a>  | <a href="https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data" target="_blank">upaya07/NeurIPS-LLM-data  </a> |  <a href="http://www.apache.org/licenses/" target="_blank">Apache License 2.0  </a> |


## Results
| Task | Score |
| ----- |------|
| MMLU - EM | 0.629 |
| MMLU - EM (Robustness) | 0.591 |
| MMLU - EM (Fairness) | 0.596 |
| MMLU Mean Win Rate | 0.417 |
| TruthfulQA - EM | 0.59 |
| TruthfulQA - EM (Robustness) | 0.541 |
| TruthfulQA - EM (Fairness) | 0.492 |
| TruthfulQA Mean Win Rate | 0.75 |
| BIG-bench - EM | 0.330 |
| BIG-bench Mean Win Rate | 0.75 |
| GSM8K - EM | 0.443 |
| GSM8K Mean Win Rate | 0.625 |
| BBQ - EM | 0.738 |
| BBQ Mean Win Rate | 0.25 |
| sam_sum - ROUGE-2 | 0.127 |
| sam_sum - Stereotypes (race) | 0.667 |
| sam_sum - Stereotypes (gender) | 0.447 |
| sam_sum - Representation (race) | 0.458 |
| sam_sum - Representation (gender) | 0.013 |
| sam_sum Mean Win Rate | 0.383 |
| corr2cause - EM | 0.615 |
| corr2cause Mean Win Rate | 0.875 |
| MATH (chain-of-thoughts) - Equivalent (chain of thought) | 0.121 |
| MATH Mean Win Rate | 0.75 |
| ethics_justice - EM | 0.68 |
| ethics_justice - EM (Robustness) | 0.645 |
| ethics_justice - EM (Fairness) | 0.62 |
| ethics_commonsense - EM | 0.41 |
| ethics_commonsense - EM (Robustness) | 0.33 |
| ethics_commonsense - EM (Fairness) | 0.345 |
| ethics_virtue - EM | 0.895 |
| ethics_virtue - EM (Robustness) | 0.865 |
| ethics_virtue - EM (Fairness) | 0.86 |
| ethics_deontology - EM | 0.63 |
| ethics_deontology - EM (Robustness) | 0.585 |
| ethics_deontology - EM (Fairness) | 0.595 |
| ethics_utilitarianism - EM | 0.72 |
| ethics_utilitarianism - EM (Robustness) | 0.6 |
| ethics_utilitarianism - EM (Fairness) | 0.645 |
| ethics Mean Win Rate | 0.55 |
| 🔥 **Score_full** | **0.579** |
| 🔥 **Score_open** | **0.516** |
| 🔥 **Score_hidden** | **0.61** |

### Top-5 Teams
| Position | Score |
| ----- |------|
| 5th rank | 0.362 |
| 4th rank | 0.371 |
| 3rd rank | 0.381 |
| 2nd rank | 0.424 |
| 🔥 **Ours (1st)** | **0.579** |

Refer to [4090_full_ranks.json](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/4090_full_ranks.json) file for scores of top-few teams that were part of final stage in competition.


## Training Data Preparation
<img width="1196" alt="Training_Data_Prep_akjindal53244" src="https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/assets/5215386/150cda6b-4b41-4fab-af45-585153c355b3">


## Birbal Models
| Birbal Models | Checkpoint | Dataset  | License|
| ----- |------| ---- | ----- |
| Birbal-200k | 🤗 <a href="https://huggingface.co/upaya07/Birbal-7B-V1" target="_blank">Birbal-200k</a>  | <a href="https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data" target="_blank">200k  </a> |  <a href="http://www.apache.org/licenses/" target="_blank">Apache License 2.0  </a> |
| Birbal-400k | 🤗 <a href="https://huggingface.co/upaya07/Birbal-400k" target="_blank">Birbal-400k</a>  | <a href="https://huggingface.co/datasets/upaya07/Birbal-400k-data" target="_blank">400k  </a> |  <a href="http://www.apache.org/licenses/" target="_blank">Apache License 2.0  </a> |
| Birbal-700k | 🤗 <a href="https://huggingface.co/upaya07/Birbal-700k" target="_blank">Birbal-700k</a>  | <a href="https://huggingface.co/datasets/upaya07/Birbal-700k-data" target="_blank">700k  </a> |  <a href="http://www.apache.org/licenses/" target="_blank">Apache License 2.0  </a> |




### Natural Instructions Dataset Preparation
[Natural Instructions](https://github.com/allenai/natural-instructions) dataset is a community effort to create a large collection of tasks and their natural language definitions/instructions. As show in above diagram, we sample from Natural Instructions dataset. Here is the 4-step process:
- Out of 1600+ tasks files, we first manually select ~450 task files relevant to the competition. **We do not use any MMLU or translation tasks.**
- A task output in Natural Instructions dataset is expected to be either an exact match or an open ended generation. Hence, we manually annotate each task file as one of two categories: Exact Match or Generation.
- We run few-shot inference on selected task files. Running few-shot inference helps with controlled generation so we can compute model performance metric quantitatively. Refer to Input and Output Schema for Mistral Inference for an example.
  - For Exact Match, we use accuracy as metric.
  - For Generation task, we use Rouge score as performance metric.
- Sampling logic: We sample ~50k examples from Generation tasks and ~50k examples from Exact Match tasks. This makes it total ~100k instances from Natural Instructions dataset.
  - For Exact match tasks: % of examples sampled from a task file depend on accuracy of that task. In general, we sample more from low-accuracy tasks and less from high-accuracy tasks. Total ~50k examples are sampled from exact match task files.
  - For Generation tasks: % of examples sampled from a task file depend on Rouge score on that task. In general, we sample more from tasks with low rouge scores. Total ~50k examples are sampled from generation task files.

#### Input and Output Schema for Mistral Inference
A record from a task file from Natural Instruction data is converted into below format. `orig_input` field is actual input without few-shot examples. `few_shot_prompt` field represents a few-shot example and is passed to Mistral-7B model for prediction. `answer` is ground truth and `prediction` is output generated by Mistral-7B base model.
```
{
  "orig_input": "Context: I sold my $90,000.00 Mercedes G500 and bought 3 Prius's, because I got tired of being pulled over by Police. #Adapt @chrisrock\u2014 Isaiah Washington (@IWashington) April 1, 2015 Question: how many prius's did they buy? Answer: three",
  "few_shot_prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIn this task, you are given a context tweet, a question and corresponding answer of given question. Your task is to classify this question-answer pair into two categories: (1) \"yes\" if the given answer is right for question, and (2) \"no\" if the given answer is wrong for question.\n\n### Input:\nContext: Our prayers are with the students, educators & families at Independence High School & all the first responders on the scene. #PatriotPride\u2014 Doug Ducey (@dougducey) February 12, 2016 Question: at which school were first responders on the scene for? Answer: arizona high school\n\n### Response:\nno\n\n### Input:\nContext: @williebosshog huge love to you/your family huge respect for your business prosperities and the family values you still all behold. big fan\u2014 Liam Payne (@Real_Liam_Payne) January 18, 2014 Question: what was liam showing towards willy? Answer: huge respect\n\n### Response:\nyes\n\n### Input:\nContext: @williebosshog huge love to you/your family huge respect for your business prosperities and the family values you still all behold. big fan\u2014 Liam Payne (@Real_Liam_Payne) January 18, 2014 Question: what was liam showing towards willy? Answer: jealousy\n\n### Response:\nno\n\n### Input:\nContext: Our prayers are with the students, educators & families at Independence High School & all the first responders on the scene. #PatriotPride\u2014 Doug Ducey (@dougducey) February 12, 2016 Question: at which school were first responders on the scene for? Answer: independence high school\n\n### Response:\nyes\n\n### Input:\nContext: I sold my $90,000.00 Mercedes G500 and bought 3 Prius's, because I got tired of being pulled over by Police. #Adapt @chrisrock\u2014 Isaiah Washington (@IWashington) April 1, 2015 Question: how many prius's did they buy? Answer: three\n\n### Response:\n",
  "answer": [
   "yes"
  ],
  "prediction": "yes\n\n### Input:\nContext: I sold my $90,000.00 Mercedes G500 and bought 3 Pri"
}
```

### Data Preparation Scripts
- [prepare_math_reasoning_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_math_reasoning_dataset.py) for preparing maths reasoning dataset.
- [prepare_exact_match_tasks_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_exact_match_tasks_dataset.py) for preparing dataset for exact match tasks from Natural Instructions.
- [prepare_generation_tasks_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_generation_tasks_dataset.py) for preparing dataset for generation tasks from Natural Instructions.
- [combine_datasets.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/combine_datasets.py) combines all datasets as shown in above diagram and prepares final train/validation splits.


**Final model training data**: https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data

## Model Training

```
# clone repository
git clone git@github.com:Upaya07/NeurIPS-llm-efficiency-challenge.git
cd NeurIPS-llm-efficiency-challenge/training/axolotl

# installation
pip install packaging
pip install -e '.[flash-attn]'
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
pip install -U git+https://github.com/huggingface/peft.git

# Downloads required data and launches model fine-tuning. Runs 3-epochs on data. Script keeps track of the best checkpoint based on eval_loss.
# nips_02.yml file contains all hyperparams.
accelerate launch -m axolotl.cli.train examples/mistral/nips/nips_02.yml
```

### Expected loss curve
![W B Chart 11_25_2023, 12_39_44 PM](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/assets/5215386/1af496f4-43d5-4f70-b666-162a8687d45d)



## Team Members
- [Ashvini Kumar Jindal](https://www.linkedin.com/in/ashvini-jindal-26653262/)
- [Ankur Parikh](https://www.linkedin.com/in/ankurnlpexpert/)
- [Pawan Rajpoot](https://www.linkedin.com/in/pawanrajpoot/)


## Anknowledgements
- We would like to thank [Akshita Sukhlecha](https://www.linkedin.com/in/akshita-sukhlecha/) for continuously helping with smooth submissions of models during competition, preparing docker files for final submissions, thoroughly testing the final model and subsequently proposing post-processing rules to process model output.
