import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,TextStreamer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_pretrained_model(model_name,max_seq_length=2048,load_in_4bit=True,local_files_only=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    return model, tokenizer

def inference(model,tokenizer,text,max_new_tokens=256):
    messages = [
        {"role": "user", "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    text_streamer = TextStreamer(tokenizer,skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(inputs, streamer = text_streamer, max_new_tokens = max_new_tokens,pad_token_id=tokenizer.eos_token_id,attention_mask=torch.ones(inputs.shape,dtype=torch.long,device=model.device))
    response=outputs[0][inputs.shape[-1]:]
    answer=tokenizer.decode(response, skip_special_tokens=True)
    return answer

def dialogue(model,tokenizer,save_log=True,filepath="dialogue/dialogue.txt"):
    question = input("\n请输入问题: ")
    answer = inference(model,tokenizer,question)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    if save_log:
        with open(filepath,"a") as f:
            f.write(f"用户: {question}\n")
            f.write(f"LLM: {answer}\n")
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-u","--use_original",action="store_true",help="-u表示使用预训练模型")
    parser.add_argument("-o","--original_model",type=str,help="预训练模型路径")
    parser.add_argument("-f","--finetune_model",type=str,help="微调模型路径")
    return parser.parse_args()

if __name__ == "__main__":
    args =parse_args()
    if args.use_original:
        original_model = "/data/nlp_share/Llama_models/Llama-3.1-8B-Instruct"
        if args.original_model:
            original_model = args.original_model
        print("使用预训练模型:",original_model)
        model,tokenizer = load_pretrained_model(original_model)
    else:
        finetune_model = "lora_model"
        if args.finetune_model:
            finetune_model = args.finetune_model
        print("使用微调模型:",finetune_model)
        model,tokenizer = load_pretrained_model(finetune_model)
    print("模型加载完成，开始对话")
    while True:
        dialogue(model,tokenizer)