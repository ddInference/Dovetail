#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=ai




# CPU-only
python -m hcsd.evaluation.gen_baseline_answer_llama2chat --ea-model-path ~/EAGLE-train/a800_layer_5_w1_silu_norm/state_49 --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /CPU-only/output_16-7-10  --bench-name humaneval --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./CPU-only/output_16-7-10.log 2>&1



#HCSD
python -m hcsd.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/EAGLE-train/a800_layer_5_w1_silu_norm/state_49 --draft-model-type HCSD --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /a800_layer_5_w1_silu_norm/output_16-7-10  --bench-name humaneval --total-token 16 --depth 7 --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./a800_layer_5_w1_silu_norm/output_16-7-10.log 2>&1

#sheared-llama-1.3B
python -m hcsd.evaluation.gen_ea_answer_llama2chat --ea-model-path ~model/sheared-llama-1.3B --draft-model-type shearedllama --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /shearedllama/output_16-7-10  --bench-name humaneval --total-token 16 --depth 7 --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./shearedllama/output_16-7-10.log 2>&1

#tinyllama
python -m hcsd.evaluation.gen_ea_answer_llama2chat --ea-model-path ~model/tinyllama-1.1B --draft-model-type tinyllama --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /tinyllama/output_16-7-10  --bench-name humaneval --total-token 16 --depth 7 --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./tinyllama/output_16-7-10.log 2>&1

#eagle
python -m hcsd.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/EAGLE-train/a800_layer_1_norm_0/state_48 --draft-model-type eagle --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /eagle/output_16-7-10  --bench-name humaneval --total-token 16 --depth 7 --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./eagle/output_16-7-10.log 2>&1


#hf-offload
python -m hcsd.evaluation.gen_baseline_answer_llama2chat_offload_hf --base-model-path ~/model/Llama-2-7b-chat-hf --model-id /hf_offload/output_16-7-10  --bench-name humaneval --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./hf_offload/output_16-7-10.log 2>&1

# partial-offload
python -m hcsd.evaluation.gen_baseline_answer_llama2chat_partial_offload --ea-model-path ~/EAGLE-train/a800_layer_5_w1_silu_norm/state_49 --draft-model-type HCSD --base-model-path ~/model/Llama-2-7b-chat-hf --ngl 14 --partial-offload --model-id /partial_offload/output_16-7-10  --bench-name humaneval --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./partial_offload/output_16-7-10.log 2>&1

# partial-offload+SD
python -m hcsd.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/EAGLE-train/a800_layer_5_w1_silu_norm/state_49 --draft-model-type HCSD --base-model-path ~/model/Llama-2-7b-chat-hf --ngl 10 --partial-offload-SD --model-id /partial-offload+SD/output_16-7-10  --bench-name humaneval --total-token 16 --depth 7 --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1 --temperature 0.0 > ./partial-offload+SD/output_16-7-10.log 2>&1

python speed.py



