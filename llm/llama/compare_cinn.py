import paddle
import numpy as np


# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_1'))

# path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_launch-out-silu_0.tmp_0_1"
# path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/cast-out-silu_0.tmp_0_1"

# (tmp_54, 
# cast_11.tmp_0, 
# tmp_52.cast_fp16, 
# rsqrt_4.tmp_0, 
# fill_constant_49.tmp_0, 
# rsqrt_0.tmp_0, 
# tmp_48, 
# linear_12.tmp_0, 
# cast_36.tmp_0, 
# fill_constant_23.tmp_0, 
# cast_16.tmp_0, 
# linear_5.tmp_0, 
# cast_6.tmp_0, 
# tmp_41, 
# cast_22.tmp_0, 
# unsqueeze2_2.tmp_0, 
# fill_constant_31.tmp_0, 
# fill_constant_25.tmp_0, 
# tmp_45.cast_fp16, 
# transpose_4.tmp_0, 
# tmp_17, 
# cast_12.tmp_0, 
# rsqrt_1.tmp_0, 
# fill_constant_33.tmp_0, 
# fill_constant_35.tmp_0, 
# fill_constant_47.tmp_0, 
# fill_constant_43.tmp_0, 
# rsqrt_2.tmp_0, 
# silu_0.tmp_0, 
# tmp_47, 
# fill_constant_37.tmp_0, 
# fill_constant_45.tmp_0, 
# tmp_30, cast_24.tmp_0, 
# transpose_12.tmp_0, 
# softmax_0.tmp_1, 
# tmp_24, tmp_39, 
# transpose_5.tmp_0, fill_constant_41.tmp_0, tmp_21.cast_fp16, cast_34.tmp_0, fill_constant_21.tmp_0, cast_28.tmp_0, silu_1.tmp_0, cast_3.tmp_0, rsqrt_3.tmp_0, unsqueeze2_3.tmp_0, cast_40.tmp_0, cast_7.tmp_0, tmp_15, tmp_23, tmp_28.cast_fp16, unsqueeze2_5.tmp_0, tmp_4.cast_fp16, reshape2_4.tmp_0, transpose_11.tmp_0, linear_11.tmp_0, unsqueeze2_4.tmp_0, tmp_6, fill_constant_29.tmp_0, softmax_1.tmp_1, reshape2_9.tmp_0, cast_18.tmp_0, cast_30.tmp_0, fill_constant_27.tmp_0, fill_constant_39.tmp_0, cast_10.tmp_0, linear_4.tmp_0, )

# Tensor(shape=[1, 2048, 1], dtype=float16, place=Place(gpu:0), stop_gradient=True,
#        [[[0.77734375],
#          [0.78857422],
#          [0.79589844],
#          ...,
#          [0.88134766],
#          [0.91601562],
#          [0.94287109]]])
# Tensor(shape=[1, 2048, 1], dtype=float16, place=Place(gpu:0), stop_gradient=True,
#        [[[0.77734375],
#          [0.78857422],
#          [0.79589844],
#          ...,
#          [0.88183594],
#          [0.91601562],
#          [0.94287109]]])

path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-output-unsqueeze2_3.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-unsqueeze2_3.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/matmul_v2-input-transpose_4.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-transpose_4.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-input-linear_0.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-linear_0.tmp_0_1"


# path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/matmul_v2-input-tmp_4_1"
# path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-tmp_4_1"

# path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/matmul_v2-output-linear_0.tmp_0"
# path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-linear_0.tmp_0_1"


# path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/transpose2-output-transpose_4.tmp_0_1"
# path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-transpose_4.tmp_0_1"

# w_comp = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_mul-input-create_parameter_0.w_0_1"
# w_cinn = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_launch-input-create_parameter_0.w_0_1"

comp_var = paddle.Tensor(paddle.base.core.load_tensor_c(path1))
cinn_var = paddle.Tensor(paddle.base.core.load_tensor_c(path2))


print(cinn_var)
print(comp_var)
print(paddle.allclose(cinn_var, comp_var))
np.testing.assert_allclose(cinn_var, comp_var, rtol=0)
# # np.testing.assert_equal(cinn_var.numpy(), comp_var.numpy())
# print(np.allclose(cinn_var, comp_var, rtol=1e-3))

# Tensor(shape=[1, 2048, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[1.65527344],
#          [1.60839844],
#          [1.57910156],
#          ...,
#          [1.28710938],
#          [1.19140625],
#          [1.12500000]]])
# Tensor(shape=[1, 2048, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[1.65527344],
#          [1.60839844],
#          [1.57812500],
#          ...,
#          [1.28613281],
#          [1.19140625],
#          [1.12500000]]])

