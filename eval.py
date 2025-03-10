import os
import json
import torch
import argparse

from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--dataset_tag", type=str)
    parser.add_argument("--max_card", type=str)

    return parser.parse_args()


def main(test_dir, dataset_tag, max_card):

    model_id = f"./models_output/Meta-Llama-3-8B-ft_{dataset_tag}_padtoken_{max_card}"
    print(model_id)
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir='./cache')
    tokenizer.padding_side = "left"

    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='cuda',
        use_cache=True,
        cache_dir='./cache',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = model.eval()

    for test_file in os.listdir(test_dir):

        test_file = os.path.join(test_dir, test_file)
        tag = test_dir.split('/')[-1]

        output_file = f"./reports/{model_id.split('/')[-1]}___{tag}___{test_file.split('/')[-1]}"
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with open(test_file, 'r', encoding='utf8') as f:
            total_lines = sum(1 for _ in f)

        with tqdm(total=total_lines) as pbar:

            with open(output_file, 'w', encoding='utf8') as f_out:
                with open(test_file, 'r', encoding='utf8') as f:

                    for l in f:

                        x = json.loads(l)

                        instruction = x["instruction"]
                        input_sentence = x["input"]

                        messages = [
                            {"role": "user", "content": instruction + " Input: \"" + input_sentence + "\""},
                        ]

                        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

                        set_seed(42)
                        outputs = model.generate(
                            input_ids.to('cuda'),
                            max_length=1024,
                            eos_token_id=terminators,
                            num_beams=1,
                            do_sample=False
                        )

                        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

                        answer = response
                        prompt = messages[0]["content"]
                        prompt_formatted = tokenizer.decode(input_ids[0], skip_special_tokens=False)

                        x["answer"] = answer
                        x["prompt"] = prompt
                        x["prompt_formatted"] = prompt_formatted

                        json.dump(x, f_out)
                        f_out.write('\n')

                        pbar.update(1)


if __name__ == "__main__":

    args = get_args()
    main(**vars(args))

