import argparse
import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from instruction_parser import InstructionParser

input_file = "/data/sh/project/Fine_Tuning/Datasets/ACL/SFT/2014-2022.json"
output_dir = "/data/sh/project/Fine_Tuning/Datasets/ACL/translator/DPO/partial/"
output_file = "/data/sh/project/Fine_Tuning/Datasets/ACL/translator/DPO/2014-2022.json"
translate_model = "/data/nlp_share/Llama_models/Llama-3.2-3B-Instruct"

def get_clients(client_num: int) -> list:
    ports = [8081 + i for i in range(client_num)]
    clients = []
    for port in ports:
        clients.append(OpenAI(api_key="0", base_url=f"http://0.0.0.0:{port}/v1"))
    return clients


def get_chinese_translation(client, input_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": InstructionParser().original_to_chinese,
        },
        {"role": "user", "content": input_text},
    ]
    result = client.chat.completions.create(messages=messages, model=translate_model)
    return result.choices[0].message.content


def get_english_translation(client, input_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": InstructionParser().chinese_to_english,
        },
        {"role": "user", "content": input_text},
    ]
    result = client.chat.completions.create(messages=messages, model=translate_model)
    return result.choices[0].message.content


def process_entry(client, entry):
    input_text = entry["input"]
    try:
        chinese_translation = get_chinese_translation(client, input_text)
        english_translation = get_english_translation(client, chinese_translation)
        return {
            "instruction": InstructionParser().translator_instruction,
            "input": chinese_translation,
            "chosen": input_text,
            "rejected": english_translation,
        }
    except Exception as e:
        print(f"处理出错: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_workers", type=int, default=12)
    parser.add_argument("-c", "--chunk_size", type=int, default=300)
    parser.add_argument("-s", "--start_index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    max_workers = args.max_workers
    chunk_size = args.chunk_size
    start_index = args.start_index

    clients = get_clients(max_workers)

    print(f"共有 {max_workers} 个客户端，每处理 {chunk_size} 条数据保存一次")

    with open(input_file, "r") as f:
        data = json.load(f)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_data = len(data)
    data = data[start_index:]
    print(
        f"共有 {total_data} 条数据，从第 {start_index} 条开始处理，共需处理 {len(data)} 条数据"
    )
    results = []

    start_time = time.time()

    chunk_number = start_index

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_entry, clients[i % max_workers], entry): entry
            for i, entry in enumerate(data)
        }

        with tqdm(total=total_data, desc="处理进度", initial=start_index) as pbar:
            for i, future in enumerate(as_completed(future_to_entry)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)

                    pbar.update(1)  # 更新进度条

                    if (i + 1) % chunk_size == 0:
                        chunk_number = i + 1
                        temp_file = f"{output_dir}{chunk_number}.json"
                        print(
                            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已处理 {chunk_number} 条数据，保存到 {temp_file}"
                        )
                        with open(temp_file, "w") as f_temp:
                            json.dump(results, f_temp, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"任务出错: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n处理完成，总耗时: {total_time:.2f} 秒")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"所有数据已保存到 {output_file}")


if __name__ == "__main__":
    main()
