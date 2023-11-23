import os
import json
import random
from datasets import load_dataset

# This path contains all the datasets: https://drive.google.com/drive/folders/1bPfxrwcgGrX3-3CHJ0SLckY22zgdPwCj
# Todo: Refer to this for description of each dataset:

"""
How to run the script:
    1. Install necessary libraries.
    2. Replace local dataset paths with datasets from above drive path
    3. Replace output paths at end of the script to save dataset at your own local paths.
"""

def fetch_instruction(task_file_name):
    with open(os.path.join("/home/minimalist/work/natural-instructions/tasks", task_file_name)) as f:
        task_json = json.load(f)
    return task_json["Definition"][0]

def prepare_quac_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/quac_train.json"
    data = []
    lines = open(file_path, "r").readlines()
    for line in lines:
        rec = json.loads(line.strip())
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) > 800:
            continue
        data.append(rec)
    random.shuffle(data)
    sampled_data = data
    print(f"quac dataset size: {len(sampled_data)}")
    return sampled_data


def prepare_openQA_files():
    data = []

    file_paths = ["/home/minimalist/Downloads/nips_training_data/openQA_train.json"
                  ]
    for file_path in file_paths:
        lines = open(file_path, "r").readlines()
        for line in lines:
            rec = json.loads(line.strip())
            rec['instruction'] = "You are given an incomplete sentence along with a few reference completions. Only one of the reference completion is coherent and correct based on the context in sentence and all other reference completions are incorrect. Your task is to choose best matching completion from reference options and return that."
            if len(f"{rec['instruction']} {rec['input']}".split(" ")) > 800:
                continue
            data.append(rec)
    random.shuffle(data)
    sampled_data = data
    print(f"OpenQA dataset size: {len(sampled_data)}")
    return sampled_data


def prepare_cnn_dailymail_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/cnn_dailymail_summarization_random_train.json"
    data = []
    lines = open(file_path, "r").readlines()
    for line in lines:
        rec = json.loads(line.strip())
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) > 800:
            continue
        data.append(rec)
    random.shuffle(data)
    sampled_data = data
    print(f"CNN/DailyMail dataset size: {len(sampled_data)}")
    return sampled_data

def prepare_maths_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/math_reasoning.json"

    with open(file_path) as f:
        data = json.load(f)

    sampled_data = []
    for rec in data:
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) < 800 and "print(" not in rec['output']:
            sampled_data.append(rec)

    random.shuffle(sampled_data)
    sampled_data = sampled_data[:250000]
    print(f"maths_reasoning dataset size: {len(sampled_data)}")
    return sampled_data


def prepare_platypus_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/platypus_no_llm.json"

    with open(file_path) as f:
        data = json.load(f)

    sampled_data = []
    for rec in data:
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) < 800:
            sampled_data.append(rec)

    random.shuffle(sampled_data)
    sampled_data = sampled_data
    print(f"platypus dataset size: {len(sampled_data)}")
    return data


def prepare_super_natural_generation_tasks_samples_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/natural_instructions_generation_tasks_dataset.json"
    task_to_instuction_map = {}

    with open(file_path) as f:
        data = json.load(f)
    print("Loaded file")

    sampled_data = []
    for i, rec in enumerate(data):
        if rec["task_file"] not in task_to_instuction_map:
            rec['instruction'] = fetch_instruction(rec["task_file"])
            task_to_instuction_map[rec["task_file"]] = rec['instruction']
        else:
            rec['instruction'] = task_to_instuction_map[rec["task_file"]]
        rec.pop('task_file', None)
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) < 800:
            sampled_data.append(rec)

    random.shuffle(sampled_data)
    sampled_data = sampled_data
    print(f"Super Natural Sentence dataset size: {len(sampled_data)}")
    return sampled_data


def prepare_super_natural_exact_match_tasks_samples_file():
    file_path = "/home/minimalist/Downloads/nips_training_data/natural_instructions_exact_match_tasks_dataset.json"

    with open(file_path) as f:
        data = json.load(f)

    random.shuffle(data)
    sampled_data = data
    print(f"Super Natural One Word dataset size: {len(sampled_data)}")
    return sampled_data



def generate_lima_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def prepare_lima_file():
    dataset = load_dataset("GAIR/lima")["train"]
    data = []
    for convo in dataset["conversations"]:
        rec = {"instruction": "", "input": convo[0], "output": convo[1]}
        if len(f"{rec['instruction']} {rec['input']}".split(" ")) < 800:
            data.append(rec)
    print(f"LIMA dataset size: {len(data)}")
    return data



data = []
data += prepare_quac_file()
data += prepare_openQA_files()
data += prepare_cnn_dailymail_file()
data += prepare_maths_file()
data += prepare_platypus_file()
data += prepare_super_natural_generation_tasks_samples_file()
data += prepare_super_natural_exact_match_tasks_samples_file()
data += prepare_lima_file()

print(f"total data size: {len(data)}")

random.shuffle(data)

instruction_prefix = "### Instruction:\n"
input_prefix = "### Input:\n"
response_prefix = "### Response:\n"
final_dataset = []
all_inputs = set()
for each in data:
    if each['input'] not in all_inputs:
        all_inputs.add(each['input'])
    else:
        continue

    if len(each['input'].split(" ")) <= 4 or len(each['output'].split(" ")) < 1:
        continue

    rec = {}
    rec['question'] = ''
    if len(each['instruction'].strip()) > 0:
        rec['question'] += f"{instruction_prefix}{each['instruction']}\n\n"
    rec['question'] += f"{input_prefix}{each['input']}\n\n"
    rec['question'] += f"{response_prefix}"
    rec['answer'] = f"{each['output']}"
    final_dataset.append(rec)

print(f"final_dataset size: {len(final_dataset)}")
train_dataset = final_dataset[:int(0.98*len(final_dataset))]
eval_dataset = final_dataset[int(0.98*len(final_dataset)):]

with open(f"/home/minimalist/Downloads/nips_training_data/train_dataset.json", 'w') as f:
    json.dump(train_dataset, f, indent=1)

with open(f"/home/minimalist/Downloads/nips_training_data/eval_dataset.json", 'w') as f:
    json.dump(eval_dataset, f, indent=1)
