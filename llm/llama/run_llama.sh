#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=1
export FLAGS_save_tensor=true
export FLAGS_cudnn_deterministc=1
export FLAGS_cinn_cudnn_deterministc=1
# export FLAGS_check_nan_inf=1
rm -rf ./save_cinn/*
export FLAGS_save_load_path="./save_cinn/"


# # 跑 动转静 + 组合算子时打开下面这1行
export FLAGS_prim_all=true
# #跑 动转静 + 组合算子 + CINN时打开下面这6行
export FLAGS_use_cinn=1
export FLAGS_use_reduce_split_pass=1
export FLAGS_nvrtc_compile_to_cubin=1
export FLAGS_cinn_use_op_fusion=1
export FLAGS_cinn_parallel_compile_size=8
result of build_cinn_pass, split paddle graph to some cinn sub-graph, use paddle op/var name
rm -rf ./cinn_graph/*
export FLAGS_cinn_subgraph_graphviz_dir="./cinn_graph/"

# before and after cinn program and graph pass(including group opfusion pass) in each sub-graph
rm -rf ./cinn_pass/*
export FLAGS_cinn_pass_visualize_dir="./cinn_pass/"

# result of cinn group, detailed each group in each sub-graph, use cinn op/var name, and generate code
rm -rf ./cinn_fusion_graph/*
export FLAGS_cinn_fusion_groups_graphviz_dir="./cinn_fusion_graph/"


task_name_or_path="llama_output"
python run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "__internal_testing__/distributed-projection-llama-7b" \
    --tokenizer_name_or_path "__internal_testing__/distributed-projection-llama-7b" \
    --input_dir "./data/" \
    --output_dir "./data/$task_name_or_path" \
    --split 940,50,10 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 512 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 1 \
    --save_steps 100 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0\
    --recompute 0 \
    --do_train \
    --do_eval \
    --device "gpu"
    #--model_name_or_path "facebook/llama-7b" \
    #--tokenizer_name_or_path "facebook/llama-7b" \
    #--sharding "stage2" \


# cinn
# cinn 
# {
#     "epoch": 0.0,
#     "train_loss": 11.24561882019043,
#     "train_runtime": 66.8144,
#     "train_samples_per_second": 0.014966823145462849,
#     "train_steps_per_second": 0.014966823145462849
# }

# {
#     "epoch": 0.0,
#     "train_loss": 11.245615005493164,
#     "train_runtime": 51.3153,
#     "train_samples_per_second": 0.01948738375118561,
#     "train_steps_per_second": 0.01948738375118561
# }
