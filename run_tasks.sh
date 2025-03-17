#!/bin/bash


TASKS=(
    "python -m dovetail.evaluation.gen_baseline_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_w1_silu_norm/state_49 --draft-model-type dovetail --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id ess-llama-2-chat-7b-baseline_quantize  --bench-name humaneval --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0"
    "python -m dovetail.evaluation.gen_ea_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_norm_0/state_48 --draft-model-type eagle --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id ess-llama-2-chat-7b_quantize_eagle_7_4_10 --bench-name humaneval --total-token 7 --depth 4  --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0"
    "python -m dovetail.evaluation.gen_ea_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_w1_silu_norm/state_49 --draft-model-type dovetail --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id ess-llama-2-chat-7b-quantize_dovetail_7_4_10 --bench-name humaneval --total-token 7 --depth 4  --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0" 
    
    "python -m dovetail.evaluation.gen_baseline_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_w1_silu_norm/state_49 --draft-model-type dovetail --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id  ess-llama-2-chat-7b-baseline_quantize --bench-name mt_bench_1 --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0"
    "python -m dovetail.evaluation.gen_ea_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_norm_0/state_48 --draft-model-type eagle --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id ess-llama-2-chat-7b_quantize_eagle_7_4_10 --bench-name mt_bench_1 --total-token 7 --depth 4  --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0"
    "python -m dovetail.evaluation.gen_ea_answer_llama2chat --ea-model-path /home/zlb/model/eagle/a800_layer_1_w1_silu_norm/state_49 --draft-model-type dovetail --base-model-path /home/zlb/model/llama-2-7B-chat-hf --model-id ess-llama-2-chat-7b-quantize_dovetail_7_4_10 --bench-name mt_bench_1 --total-token 7 --depth 4  --top-k 10 --num-gpus-per-model 1 --num-gpus-total 1  --temperature 0.0"

)
for index in "${!TASKS[@]}"; do
    task_num=$((index + 1))
    echo "------------------------------------------------"
    echo "Running task $task_num/${#TASKS[@]}..."
    echo "Start: $(date)"
    echo "------------------------------------------------"
    
    
    command="${TASKS[index]}"
    ea_model_path=$(echo "$command" | grep -o -- '--ea-model-path [^ ]*' | cut -d' ' -f2)
    
    
    eval "$command"
    
    
    echo "Checking zombie processes..."
    if pkill -f "$ea_model_path"; then
        echo "Killed processes with pattern: $ea_model_path"
    else
        echo "No processes found with pattern: $ea_model_path"
    fi
    
    remaining=$(pgrep -f "$ea_model_path" | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo "WARNING: Found $remaining residual processes, force killing..."
        pkill -9 -f "$ea_model_path"
    fi
    
    echo "------------------------------------------------"
    echo "Task $task_num completed"
    echo "End: $(date)"
    echo "------------------------------------------------"
    echo
done

echo "All tasks finished!"
