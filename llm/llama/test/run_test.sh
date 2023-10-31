export CUDA_VISIBLE_DEVICES=6
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=1
export FLAGS_save_tensor=true
export FLAGS_cudnn_deterministc=1
export FLAGS_cinn_cudnn_deterministc=1
# export FLAGS_check_nan_inf=1
rm -rf ./save_cinn/*
export FLAGS_save_load_path="./save_cinn/"

# 跑 动转静 + 组合算子时打开下面这1行
export FLAGS_prim_all=true
# 跑 动转静 + 组合算子 + CINN时打开下面这6行
export FLAGS_use_cinn=1
export FLAGS_use_reduce_split_pass=1
export FLAGS_nvrtc_compile_to_cubin=1
export FLAGS_cinn_use_op_fusion=1
export FLAGS_cinn_parallel_compile_size=8
# result of build_cinn_pass, split paddle graph to some cinn sub-graph, use paddle op/var name
rm -rf ./cinn_graph/*
export FLAGS_cinn_subgraph_graphviz_dir="./cinn_graph/"

# before and after cinn program and graph pass(including group opfusion pass) in each sub-graph
rm -rf ./cinn_pass/*
export FLAGS_cinn_pass_visualize_dir="./cinn_pass/"

# result of cinn group, detailed each group in each sub-graph, use cinn op/var name, and generate code
rm -rf ./cinn_fusion_graph/*
export FLAGS_cinn_fusion_groups_graphviz_dir="./cinn_fusion_graph/"


task_name_or_path="llama_output"
GLOG_v=6 python test_element_mul.py 