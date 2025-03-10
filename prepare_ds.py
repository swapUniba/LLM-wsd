import json
import string
import argparse
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tag", type=str)
    parser.add_argument("--max_card", type=str)

    return parser.parse_args()


def tokenize_input(x, tokenizer):

    output = x["output"]
    instruction = x["instruction"]
    input_string = x["input"]

    try:
        _ = str(int(output))
        task_type = "multiple_choice"
    except:
        task_type = "generation"

    if not output[-1] in string.punctuation:
        output += "."

    if not instruction[-1] in string.punctuation:
        instruction += "."

    if not input_string[-1] in string.punctuation:
        input_string += "."

    messages = [
        {"role": "user", "content": instruction + " Input: \"" + input_string + "\""},
        {"role": "assistant", "content": output}
    ]

    inputs = tokenizer.apply_chat_template(messages)
    inputs_not_tokenized = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"tokenized_len": len(inputs), "not_tokenized": inputs_not_tokenized, "task_type": task_type}


def main(dataset_tag, max_card):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir="./cache")

    ds = load_dataset("json", data_files={"train": f"./data/train_xlwsd_{dataset_tag}.jsonl"})["train"]

    print(len(ds))
    ds = ds.map(lambda x: tokenize_input(x, tokenizer), num_proc=4)
    ds = ds.shuffle(42)

    print(pd.DataFrame.from_dict({'token_len': ds["tokenized_len"]}).describe(percentiles=[.95]))

    max_length = 512

    gen_instances_dict = {lang: 0 for lang in ["en", "es", "de", "it", "fr"]}
    choice_instances_dict = {lang: 0 for lang in ["en", "es", "de", "it", "fr"]}

    with open(f'./dataset_{dataset_tag}_final_8B_{str(max_card)}.jsonl', 'w', encoding='utf8') as f:

        for x in tqdm(ds):

            if x["tokenized_len"] > max_length:
                continue
            
            if x["task_type"] == "generation":
                task_dict = gen_instances_dict
            else:
                task_dict = choice_instances_dict
            
            total_instances = task_dict[x["lang"]]

            if total_instances == max_card:
                continue
                
            task_dict[x["lang"]] += 1
            
            json.dump(x, f)
            f.write('\n')

    ds = load_dataset("json", data_files={"train": f"./dataset_{dataset_tag}_final_8B_{str(max_card)}.jsonl"})["train"]

    print(ds[0])
    print(len(ds))
    print(gen_instances_dict)
    print(choice_instances_dict)


if __name__ == "__main__":

    args = get_args()
    main(**vars(args))
