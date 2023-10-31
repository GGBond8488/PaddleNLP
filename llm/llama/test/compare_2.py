import paddle
import numpy as np


cinn_q_embed = paddle.load("cinn_q_embed.pdtensor")
cinn_k_embed = paddle.load("cinn_k_embed.pdtensor")
cinn_fusion_q_embed = paddle.load("cinn_fusion_q_embed.pdtensor")
cinn_fusion_k_embed = paddle.load("cinn_fusion_k_embed.pdtensor")
no_cinn_q_embed = paddle.load("no_cinn_q_embed.pdtensor")
no_cinn_k_embed = paddle.load("no_cinn_k_embed.pdtensor")


np.testing.assert_allclose(cinn_q_embed, no_cinn_q_embed, rtol=0)
np.testing.assert_allclose(cinn_k_embed, no_cinn_k_embed, rtol=0)
np.testing.assert_allclose(cinn_fusion_k_embed, no_cinn_k_embed, rtol=0)
np.testing.assert_allclose(cinn_fusion_q_embed, no_cinn_q_embed, rtol=0)

# cinn_index_path = "/root/paddlejob/PaddleNLP/llm/llama/test/save_cinn/cinn_instruction_run-output-unsqueeze2_1.tmp_0_1"
# comp_index_path = "/root/paddlejob/PaddleNLP/llm/llama/test/save_comp/unsqueeze2-output-unsqueeze2_1.tmp_0_1"


# cinn_var = paddle.Tensor(paddle.base.core.load_tensor_c(cinn_index_path))
# comp_var = paddle.Tensor(paddle.base.core.load_tensor_c(comp_index_path))

# # print(cinn_var)
# # print(comp_var)
# np.testing.assert_allclose(cinn_var, comp_var, rtol=0)