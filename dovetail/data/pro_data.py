
# import json

# # 文件路径设置
# input_file = '/gf3/home/lzq/EAGLE-train/eagle/data/mt_bench/question.jsonl'  # 输入文件路径
# output_file = '/gf3/home/lzq/EAGLE-train/eagle/data/mt_bench_1/question.jsonl'  # 输出文件路径

# processed_data = []

# # 逐行读取 JSONL 文件并处理
# with open(input_file, 'r', encoding='utf-8') as infile:
#     for line in infile:
#         # 加载每一行的 JSON 数据
#         entry = json.loads(line.strip())
        
#         # 仅保留 "turns" 列表中的第一个项
#         if "turns" in entry and len(entry["turns"]) > 0:
#             entry["turns"] = [entry["turns"][0]]
        
#         # 将处理后的条目添加到列表中
#         processed_data.append(entry)

# # 将处理后的数据写入新的 JSONL 文件
# with open(output_file, 'w', encoding='utf-8') as outfile:
#     for entry in processed_data:
#         json.dump(entry, outfile, ensure_ascii=False)
#         outfile.write('\n')

# print("数据处理完成，保存至:", output_file)


import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
###########################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
############################
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
question_file = "/gf3/home/zlb/EAGLE-train/eagle/data/mt_bench/question.jsonl"
questions = load_questions(question_file,0,80)


# Assuming `questions` is your input list
output_data = []

for idx, question in enumerate(tqdm(questions)):
    # Create the conversation template and system message
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p

    # Append the user's message to the conversation
    qs = question["turns"][0]
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    # Generate the prompt
    prompt = conv.get_prompt() + " "

    # Append to output_data with index and prompt format
    output_data.append([idx, prompt])

# Save to JSON file
output_file_path = "/gf3/home/zlb/specexec-main_test/data/mtbench1_prompts.json"
with open(output_file_path, "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Prompts have been saved to {output_file_path}")
