import os
import json
import random
import math

total_rec, incorrect = 0, 0

total_records_to_sample = 50*1000
output_dir = "/home/minimalist/Downloads/nips_training_data/NI/1_word/incorrect_preds"

def fetch_instruction(task_file_name):
    # save json files locally from https://github.com/allenai/natural-instructions/tree/master/tasks
    with open(os.path.join("/home/minimalist/work/natural-instructions/tasks", task_file_name)) as f:
        task_json = json.load(f)
    return task_json["Definition"][0]

def extract_answer(answer):
    return answer.split("\n\n")[0]

def process_file(file_path):
    global total_rec, incorrect
    incorrect_predictions = []
    with open(file_path) as f:
        data = json.load(f)
    for rec in data:
        ground_truth = rec["ground_truth"]
        prediction = extract_answer(rec["prediction"])
        is_correct = False

        out_rec = {}
        out_rec['instruction'] = fetch_instruction(rec["task_file"])
        out_rec['input'] = rec["orig_input"]
        out_rec['output'] = ground_truth[0]

        if any([x == prediction for x in ground_truth]):
            is_correct = True

        if not is_correct:
            incorrect_predictions.append(out_rec)

    random.shuffle(incorrect_predictions)

    task_accuracy = (len(data)-len(incorrect_predictions)) / len(data)

    total_rec += len(data)
    incorrect += len(incorrect_predictions)
    file_name = file_path.split("/home/minimalist/Downloads/super_NI_mistral_inference_output/1_word/")[1]
    print(f"\n{file_name}: #task_records {len(data)}, Incorrect: {1-task_accuracy}\nOVERALL: #total_records: {total_rec}, total_incorrect: {incorrect} Incorect: {incorrect / total_rec}")

    output = {}
    output['source'] = file_name
    output["accuracy"] = task_accuracy
    output["instances"] = incorrect_predictions

    with open(f"{output_dir}/{file_name}", 'w') as f:
        json.dump(output, f, indent=1)


def find_incorrect_predictions():
    files = os.listdir("/home/minimalist/Downloads/super_NI_mistral_inference_output/1_word/")
    for file in files:
        if not os.path.exists(f"{output_dir}/{file}"):
            process_file(os.path.join("/home/minimalist/Downloads/super_NI_mistral_inference_output/1_word", file))
        else:
            print(f"Path already exists! {output_dir}/{file}")

def sample_by_accuracy(file_path, multiplication_factor = 1.0):
    with open(file_path) as f:
        data = json.load(f)

    accuracy = data["accuracy"]
    print(f"accuracy: {accuracy}")
    instances: object = data["instances"]
    print(f"original #records: {len(instances)}")
    instances = [rec for rec in instances if len(f"{rec['instruction']} {rec['input']}".split(" ")) < 800]
    print(f"#records after removing long inputs: {len(instances)}")

    bucket_id = math.floor(int((accuracy*100)/10))
    if bucket_id == 0:
        samples = int(350*multiplication_factor)
    elif bucket_id == 1:
        samples = int(350*multiplication_factor)
    elif bucket_id == 2:
        samples = int(300*multiplication_factor)
    elif bucket_id == 3:
        samples = int(300*multiplication_factor)
    elif bucket_id == 4:
        samples = int(250*multiplication_factor)
    elif bucket_id == 5:
        samples = int(250*multiplication_factor)
    elif bucket_id == 6:
        samples = int(200*multiplication_factor)
    elif bucket_id == 7:
        samples = int(150*multiplication_factor)
    elif bucket_id == 8:
        samples = int(100*multiplication_factor)
    else:
        samples = int(50 * multiplication_factor)

    random.shuffle(instances)
    sampled_data = instances[:samples]
    return sampled_data


def prepare_training_data(multiplication_factor = 1.0):
    training_data = []
    data_dir = "/home/minimalist/Downloads/nips_training_data/NI/1_word/incorrect_preds/"
    files = os.listdir(data_dir)
    for file in files:
        print(f"file: {file}")
        sampled_data = sample_by_accuracy(os.path.join(data_dir, file), multiplication_factor=multiplication_factor)
        print(f"#Sampled Records: {len(sampled_data)}")

        training_data.extend(sampled_data)
        print(f"Total data so far: {len(training_data)}\n\n")

    with open(f"/home/minimalist/Downloads/nips_training_data/super_natural_1_word_data_{str(int(multiplication_factor))}.json", 'w') as f:
        json.dump(training_data, f, indent=1)


find_incorrect_predictions()
prepare_training_data()
