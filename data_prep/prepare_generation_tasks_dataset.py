from rouge import Rouge
import json
import os
import csv
import random
import math

filtered_list = []
def compute_rouge_score(llm_json, output_json):
    f = open(llm_json)
    data = json.load(f)
    rouge = Rouge()
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    final_list = []

    for i in data:
        ground_truths = i["ground_truth"]
        prediction = i["prediction"].split("\n\n")[0].strip()
        if len(ground_truths) > 0:
            rouge_scores = []
            for ground_truth in ground_truths:
                ground_truth = ground_truth.replace('.','')
                prediction = prediction.replace('.','')
                if len(ground_truth) == 0 and len(prediction) == 0:
                    ground_truth = "DUMMY"
                    prediction = "DUMMY"
                if len(ground_truth) == 0:
                    ground_truth = prediction
                if len(prediction) == 0:
                    prediction = ground_truth
                scores = rouge.get_scores(prediction.lower(), ground_truth.lower())
                score = float(scores[0]['rouge-l']['f'])
                rouge_scores.append(score)
            max_rouge_score = max(rouge_scores)
            if max_rouge_score < 0.2:
                list_2.append(i)
            if max_rouge_score >= 0.2 and max_rouge_score < 0.3:
                list_3.append(i)
            if max_rouge_score >= 0.3 and max_rouge_score < 0.4:
                list_4.append(i)
            if max_rouge_score >= 0.4 and max_rouge_score < 0.5:
                list_5.append(i)
            if max_rouge_score >= 0.5 and max_rouge_score < 0.6:
                list_6.append(i)
            if max_rouge_score >= 0.6 and max_rouge_score < 0.7:
                list_7.append(i)
            if max_rouge_score >= 0.7 and max_rouge_score < 0.8:
                list_8.append(i)

    random.shuffle(list_2)
    random.shuffle(list_3)
    random.shuffle(list_4)
    random.shuffle(list_5)
    random.shuffle(list_6)
    random.shuffle(list_7)
    random.shuffle(list_8)

    list_2_cnt = len(list_2)
    list_3_cnt = len(list_3)
    list_4_cnt = len(list_4)
    list_5_cnt = len(list_5)
    list_6_cnt = len(list_6)
    list_7_cnt = len(list_7)
    list_8_cnt = len(list_8)

    list_2_instance_cnt = int(math.floor(0.8 * list_2_cnt))
    list_3_instance_cnt = int(math.floor(0.2 * list_3_cnt))
    list_4_instance_cnt = int(math.floor(0.2 * list_4_cnt))
    list_5_instance_cnt = int(math.floor(0.2 * list_5_cnt))
    list_6_instance_cnt = int(math.floor(0.2 * list_6_cnt))
    list_7_instance_cnt = int(math.floor(0.2 * list_7_cnt))
    list_8_instance_cnt = int(math.floor(0.2 * list_8_cnt))

    for i in range(0,list_2_instance_cnt):
        final_list.append(list_2[i])
    for i in range(0,list_3_instance_cnt):
        final_list.append(list_3[i])
    for i in range(0,list_4_instance_cnt):
        final_list.append(list_4[i])
    for i in range(0,list_5_instance_cnt):
        final_list.append(list_5[i])
    for i in range(0,list_6_instance_cnt):
        final_list.append(list_6[i])
    for i in range(0,list_7_instance_cnt):
        final_list.append(list_7[i])
    for i in range(0,list_8_instance_cnt):
        final_list.append(list_8[i])

    for i in final_list:
        ground_truths = i["ground_truth"]
        input_text = i["orig_input"]
        words = input_text.split(" ")
        if len(ground_truths) == 1 and len(words) <= 1100:
            del i["id"]
            del i["few_shot_prompt"]
            del i["prediction"]
            del i["orig_output"]
            i["output"] = i["ground_truth"][0]
            i["input"] = i["orig_input"]
            del i["ground_truth"]
            del i["orig_input"]
            filtered_list.append(i)
    list_2.clear()
    list_3.clear()
    list_4.clear()
    list_5.clear()
    list_6.clear()
    list_7.clear()
    list_8.clear()

    size = len(filtered_list)
    final_list.clear()
    return size

fields = ["filename", "total"]
rows = []
total_cnt = 0
directory = '/home/utilizeai/Documents/NIPS-LLM/V2_super_natural_inference_mistral_7B'
output_directory = '/home/utilizeai/Documents/NIPS-LLM/final'
for filename in os.listdir(directory):
    print(filename)
    llm_json = os.path.join(directory, filename)
    output_json = os.path.join(output_directory, filename)
    cnt = compute_rouge_score(llm_json, output_json)
    print(cnt)
    total_cnt = total_cnt + cnt
    row = []
    row.append(filename)
    row.append(cnt)
    rows.append(row)

print(total_cnt)
filename = "/home/utilizeai/Documents/NIPS-LLM/supernatural_instructions_filter_cnt.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)

output_json = os.path.join(output_directory, "natural_instructions_generation_tasks_dataset.json")
with open(output_json, 'w') as f:
    json.dump(filtered_list, f, indent=4)
