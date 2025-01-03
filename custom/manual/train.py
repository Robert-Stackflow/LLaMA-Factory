import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

roleplay_url = "/data/sh/project/Fine_Tuning/Datasets/Classical-Chinese-Roleplay.jsonl"
ner_url = "/data/sh/project/Fine_Tuning/Datasets/HiNER-original"

def load_pretrained_model(model_name,max_seq_length=2048,load_in_4bit=True,local_files_only=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        local_files_only = True,
        offload_buffers=True
    )
    return model, tokenizer

def load_sft_dataset(train_url,type="json"):
    dataset = load_dataset(type, data_files = {"train" : train_url}, split = "train")
    dataset = dataset.map(format_prompt,batched=True)
    return dataset

def format_prompt(examples,eos_token):
    instructions = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"问题：{instruction},回答：{output}" + eos_token
        texts.append(text)
    return { "text" : texts, }
pass

def get_peft_model(model,max_seq_length=2048):
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model

def get_trainer(model,dataset,tokenizer,max_seq_length=2048):
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        tokenizer = tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )
    return trainer

if __name__ == "__main__":
    model, tokenizer = load_pretrained_model("/data/nlp_share/Llama_models/Llama-3.1-8B")
    model = get_peft_model(model)
    dataset = load_sft_dataset(ner_url)
    trainer= get_trainer(model,dataset,tokenizer)

    trainer.train()

    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")