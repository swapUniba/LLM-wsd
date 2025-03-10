import os
import torch
import datasets
import argparse

from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import set_seed


def print_on_main(to_print):
    if os.environ.get('RANK', '0') == '0' and os.environ.get('LOCAL_RANK', '0') == '0':
        print(to_print)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tag", type=str)
    parser.add_argument("--max_card", type=str)

    return parser.parse_args()


def train(dataset_tag, max_card):

    parameters = "8"
    use_lora = False

    model_id = f"meta-llama/Meta-Llama-3.1-{parameters}B-Instruct"

    if use_lora:
        new_model_name = f"models_output/Meta-Llama-3-{parameters}B-ft_{dataset_tag}_padtoken_lora_{max_card}"
    else:
        new_model_name = f"models_output/Meta-Llama-3-{parameters}B-ft_{dataset_tag}_padtoken_{max_card}"
    
    print("*" * 8)
    print(new_model_name)
    print("*" * 8)

    max_seq_length = 512
    num_train_epochs = 1
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 8
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 4e-5
    weight_decay = 0.0
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.0
    save_total_limit = 5
    logging_steps = 1

    group_by_length = True
    packing = False

    seed = 42

    print_on_main("Training parameters detected! Setting up")

    os.makedirs(new_model_name, exist_ok=True)
    output_dir = os.path.join(new_model_name, "checkpoints")

    training_config = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        save_strategy = "epoch",
        save_total_limit = save_total_limit,
        weight_decay=weight_decay,
        fp16 = False,
        bf16 = True,
        tf32 = True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=group_by_length,
        report_to = None,
    )

    set_seed(seed)

    print_on_main(training_config)
    print_on_main("LOADING DATASET")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir='./cache')
    tokenizer.pad_token = '<|finetune_right_pad_id|>'

    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
    dataset = load_dataset('json', data_files={'train': f"./dataset_{dataset_tag}_final_{parameters}B_{max_card}.jsonl"}, cache_dir="cache")

    print(f"Processed cardinality: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,
        use_cache=not gradient_checkpointing,
        cache_dir='cache',
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    if use_lora:

        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules='all-linear',
            bias="none", 
            lora_dropout=0.05,  
            task_type="CAUSAL_LM",
        )
    
    else:

        lora_config = None
                                
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        packing=packing,
        peft_config=lora_config,
        dataset_num_proc=32,
        dataset_batch_size=5000,
        dataset_text_field="not_tokenized",
        max_seq_length=max_seq_length,
        args=training_config,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    print_on_main(trainer.accelerator.state.distributed_type)

    if trainer.is_fsdp_enabled:
        print_on_main(trainer.accelerator.state.fsdp_plugin)

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    tokenizer.save_pretrained(new_model_name)
    trainer.save_model(new_model_name)


if __name__ == "__main__":

    args = get_args()
    train(**vars(args))
