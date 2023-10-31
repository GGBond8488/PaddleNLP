import paddle
import numpy as np


# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_1'))

# path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_launch-out-silu_0.tmp_0_1"
# path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/cast-out-silu_0.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-output-unsqueeze2_3.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-unsqueeze2_3.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/matmul_v2-input-transpose_4.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-transpose_4.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_mul-output-tmp_6_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-tmp_6_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/matmul_v2-output-linear_0.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-linear_0.tmp_0_1"


path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/transpose2-output-transpose_2.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-transpose_2.tmp_0_1"

path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/slice-input-eager_tmp_2_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-input-eager_tmp_2_1"

path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-output-unsqueeze2_2.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-unsqueeze2_2.tmp_0_1"




path1 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/scale-input-transpose_2.tmp_0_1"
path2 = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-transpose_2.tmp_0_1"

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

