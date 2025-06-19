import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("model/vicuna-13b-v1.3")
jsonl_file = "humaneval-temperature-0.0.jsonl"
jsonl_file_base = "humaneval-base-temperature-0.0.jsonl"

print(jsonl_file)
print(jsonl_file_base)
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



ids_total=0
ids_token_total=0


speeds=[]
speeds1=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    ids=sum(datapoint["choices"][0]['idxs'])
    ids_total += ids
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    ids_token_total += tokens
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)
    speeds1.append(datapoint["choices"][0]['new_tokens'][0]/datapoint["choices"][0]['wall_time'][0])
print('accept',ids_token_total/ids_total)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)




total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



print('speeds1',np.array(speeds1).mean())
print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())



