# NeurIPS-llm-efficiency-challenge
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Model Weight License](https://img.shields.io/badge/Model%20Weights%20License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

[Our model](https://huggingface.co/upaya07/model_02) won 🏆 first prize 🏆 in RTX 4090 track in [NeurIPS Large Language Model Efficiency Challenge: 1 LLM + 1GPU + 1Day](https://llm-efficiency-challenge.github.io/) competition. We used [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) as a base model and used QLoRA to fine-tune it for 24 hours on a single RTX 4090 GPU.
<br>

| Model | Checkpoint | Dataset  | License|
| ----- |------| ---- | ----- |
| Model | 🤗 <a href="https://huggingface.co/upaya07/model_02" target="_blank">HF Link</a>  | <a href="https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data" target="_blank">upaya07/NeurIPS-LLM-data  </a> |  <a href="http://www.apache.org/licenses/" target="_blank">Apache License 2.0  </a> |


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


### Data Preparation Scripts
- [prepare_math_reasoning_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_math_reasoning_dataset.py) for preparing maths reasoning dataset.
- [prepare_exact_match_tasks_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_exact_match_tasks_dataset.py) for preparing dataset for exact match tasks from Natural Instructions.
- [prepare_generation_tasks_dataset.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/prepare_generation_tasks_dataset.py) for preparing dataset for generation tasks from Natural Instructions.
- [combine_datasets.py](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/data_prep/combine_datasets.py) combines all datasets as shown in above diagram and prepares final train/validation splits.


**Final model training data**: https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data

For more details on fine-tuning data preparation, visit [this doc](https://docs.google.com/document/d/1FJm7IRdpjPrN3-zGSw4GuqK7MYuRJVRcYXg2252lI04/edit?usp=sharing)


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
- [Ankur Parikh](https://www.linkedin.com/in/ankurnlpexpert/)
- [Ashvini Kumar Jindal](https://www.linkedin.com/in/ashvini-jindal-26653262/)
- [Pawan Rajpoot](https://www.linkedin.com/in/pawanrajpoot/)


## Anknowledgements
- We would like to thank [Akshita Sukhlecha](https://www.linkedin.com/in/akshita-sukhlecha/) for continuously helping with smooth submissions of models during competition, preparing docker files for final submissions, thoroughly testing the final model and subsequently proposing post-processing rules to process model output.
