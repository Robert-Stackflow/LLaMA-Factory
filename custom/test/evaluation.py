import json
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,TextStreamer
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(reference, candidate):
    reference_tokens = [ref.split() for ref in reference]
    candidate_tokens = candidate.split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth_fn)

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference[0], candidate)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def calculate_exact_match(reference, candidate):
    return int(reference[0].strip() == candidate.strip())

def evaluate_batch_bleu(references, candidates):
    total_bleu, total_rouge1, total_rougeL, total_em = 0, 0, 0, 0
    for ref, cand in zip(references, candidates):
        total_bleu += calculate_bleu([ref], cand)
        rouge1, rougeL = calculate_rouge([ref], cand)
        total_rouge1 += rouge1
        total_rougeL += rougeL
        total_em += calculate_exact_match([ref], cand)
    n = len(references)
    return {
        "BLEU": total_bleu / n,
        "ROUGE-1": total_rouge1 / n,
        "ROUGE-L": total_rougeL / n,
        "EM": total_em / n,
    }

def evaluate_batch_accuracy(references, candidates):
    total = len(references)
    correct = 0
    for ref, cand in zip(references, candidates):
        if ref == cand:
            correct += 1
    return correct / total

def inference(client,instruction,text):
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": text},
    ]
    result = client.chat.completions.create(messages=messages,model='text-davinci-003')
    return result.choices[0].message.content

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_path",type=str,help="数据路径")
    parser.add_argument("-p", "--port", type=str, default=None,help="模型API端口")
    parser.add_argument("-m","--max_samples",type=int,default=1000,help="最大样本数")
    parser.add_argument("-s","--step_samples",type=int,default=10,help="每step_samples个样本评估一次")
    parser.add_argument("-o","--offset",type=int,default=0,help="数据偏移量")
    parser.add_argument("-r","--res_dir",type=str,default=None,help="结果保存路径")
    return parser.parse_args()

def save_batch(res,res_dir):
    with open(os.path.join(res_dir,"res.json"),"w") as f:
        json.dump(res,f,indent=4)

def inference_batch(data_path,client,max_samples,res_dir,offset=0):
    with open(data_path,"r") as f:
        data= json.load(f)
        res=[]
        references = []
        candidates = []
        for i in range(offset,offset+max_samples):
            instruction = data[i]["instruction"]
            input = data[i]["input"]
            reference = data[i]["output"]
            candidate = inference(client,instruction,input)
            references.append(reference)
            candidates.append(candidate)
            res.append({
                'instruction':instruction,
                'input':input,
                "reference":reference,
                "candidate":candidate
            })
            print(f"样本{i}的推理结果: ",candidate)
        save_batch(res,res_dir)
    return references,candidates,res

if __name__ == "__main__":
    args =parse_args()
    port = args.port
    max_samples = args.max_samples
    step_samples = args.step_samples
    offset = args.offset
    data_path = args.data_path
    res_dir = args.res_dir
    
    if port is None:
        print("请提供模型API端口")
        exit()

    print("初始化客户端...")
    client = OpenAI(api_key="0", base_url=f"http://0.0.0.0:{port}/v1")
    print("开始评估微调模型...")
    references,candidates,res = inference_batch(data_path,client,max_samples,res_dir)
    print("微调模型评估结果: ")
    for i in range(0,max_samples,step_samples):
        print(f"样本{i}到样本{i+step_samples}的评估结果: ",evaluate_batch_accuracy(references[i:i+step_samples],candidates[i:i+step_samples]))