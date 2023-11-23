import json
import random
from datasets import load_dataset
from difflib import SequenceMatcher

lower_case_prob = 0.2

def modify_input(question):
    num = random.randint(1, 10)
    if num <= 3:
        question = question.lower()
    return question

def remove_hash(answer: str):
    if "####" in answer:
        return answer[:answer.rindex("####")].strip()
    return answer

def format_response(answer: str, answer_identifier: str):
    answer_prefix_len = len(answer_identifier)
    if answer_identifier in answer:
        answer_prefix_start_idx = answer.index(answer_identifier)
        reasoning = remove_hash(answer[:answer_prefix_start_idx].strip())

        # ==== Enable it if we want to add "answer" as part of output
        answer = answer[answer_prefix_start_idx:].strip()
        assert len(answer) > 0
        # answer = "Answer: " + answer
        return f"{reasoning}\n{answer.strip()}"
    else:
        return answer

new_math_recs = []
valid_records = 0

math_instruct_dataset = load_dataset("TIGER-Lab/MathInstruct", "train")
valid_sources = ['data/CoT/gsm_train.json', 'data/CoT/aqua_rat.json', 'data/CoT/MATH_train.json']
print(f"math_instruct_dataset size: {len(math_instruct_dataset['train'])}")
for each in math_instruct_dataset["train"]:

    if each['source'] in valid_sources:
        output = {}
        output['instruction'] = ""
        output['input'] = modify_input(each['instruction']).strip()
        output['output'] = format_response(each['output'], "The answer is:").strip()

        new_math_recs.append(output)

        valid_records += 1

print(valid_records)

instruction_prefix = "### Instruction:\n"
input_prefix = "### Input:\n"
response_prefix = "### Response:\n"
new_math_dataset = []
for each in new_math_recs:
    if len(each['input'].split(" ")) <= 4 or len(each['output'].split(" ")) < 1:
        continue

    rec = {}
    rec['question'] = ''
    if len(each['instruction'].strip()) > 0:
        rec['question'] += f"{instruction_prefix}{each['instruction']}\n\n"
    rec['question'] += f"{input_prefix}{each['input']}\n\n"
    rec['question'] += f"{response_prefix}"
    rec['answer'] = f"{each['output']}"
    new_math_dataset.append(rec)

print(f"new_math_dataset size: {len(new_math_dataset)}")
random.shuffle(new_math_dataset)
sampled_dataset = new_math_dataset[:50000]

with open('/Users/ajindal/Downloads/math_reasoning.json', 'w') as f:
    json.dump(sampled_dataset, f, indent=1)
