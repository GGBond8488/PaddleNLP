import paddle
import numpy as np



a_cinn = paddle.load("cinn.pdtensor")
a_emb_cinn = paddle.load("cinn_emb.pdtensor")
b_no_cinn = paddle.load("no_cinn.pdtensor")
b_no_cinn_emb = paddle.load("no_cinn_emb.pdtensor")

# np.testing.assert_allclose(a_cinn, b_no_cinn, rtol=0)

# exit(0)

# mean_0 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/cast-output-mean_0.tmp_0_1"
# mean_0 = paddle.Tensor(paddle.base.core.load_tensor_c(mean_0))

# c_mean = "/root/paddlejob/PaddleNLP/llm/llama/test/save_comp/scale-input-mean_0.tmp_0_1"
# c_mean = paddle.Tensor(paddle.base.core.load_tensor_c(c_mean))

embedding_0_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/lookup_table_v2-output-embedding_0.tmp_0_1"
embedding_0 = paddle.Tensor(paddle.base.core.load_tensor_c(embedding_0_path))

tmp_4_cinn = "/root/paddlejob/PaddleNLP/llm/llama/save_cinn/cinn_instruction_run-output-tmp_4_1"
tmp_4_cinn = paddle.Tensor(paddle.base.core.load_tensor_c(tmp_4_cinn))

tmp_4_no_cinn = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_mul-output-tmp_4_1"
tmp_4_no_cinn = paddle.Tensor(paddle.base.core.load_tensor_c(tmp_4_no_cinn))

pow_0_tmp_0 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_pow-output-elementwise_pow_0.tmp_0_1"
pow_0_tmp_0 = paddle.Tensor(paddle.base.core.load_tensor_c(pow_0_tmp_0))

print(pow_0_tmp_0)
pow_0_tmp_0 = paddle.cast(pow_0_tmp_0, "float16")

# np.testing.assert_allclose(embedding_0, b_no_cinn_emb, rtol=0)
np.testing.assert_allclose(embedding_0, b_no_cinn_emb, rtol=0)
pow_ret = paddle.pow(embedding_0, 2)
np.testing.assert_allclose(pow_0_tmp_0, pow_ret, rtol=0)

# print(pow_ret)
# print(pow_0_tmp_0)

np.testing.assert_allclose(tmp_4_cinn, tmp_4_no_cinn, rtol=0)

# np.testing.assert_allclose(a_cinn, b_no_cinn, rtol=0)
# Mismatched elements: 829 / 8388608 (0.00988%)
# Max absolute difference: 0.001953
# Max relative difference: 0.0009727
#  x: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.391  , ...,  2.861  , -0.8857 ,...
#  y: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.391  , ...,  2.861  , -0.8857 ,...
# np.testing.assert_allclose(a_cinn, tmp_6_cinn, rtol=0)
# Mismatched elements: 2110448 / 8388608 (25.2%)
# Max absolute difference: 0.003906
# Max relative difference: 0.01
#  x: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.391  , ...,  2.861  , -0.8857 ,...
#  y: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.39   , ...,  2.861  , -0.8853 ,...


#np.testing.assert_allclose(tmp_6_comp, b_no_cinn)

# Mismatched elements: 3 / 8388608 (3.58e-05%)
# Max absolute difference: 0.003906
# Max relative difference: 0.0101
#  x: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.39   , ...,  2.861  , -0.8853 ,...
#  y: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.391  , ...,  2.861  , -0.8857 ,...

# np.testing.assert_allclose(a_cinn, tmp_6_cinn)
# Mismatched elements: 3 / 8388608 (3.58e-05%)
# Max absolute difference: 0.003906
# Max relative difference: 0.01
#  x: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.391  , ...,  2.861  , -0.8857 ,...
#  y: array([[[ 0.2668 ,  0.798  , -0.7896 , ...,  1.306  ,  1.159  ,
#           1.838  ],
#         [-0.3386 ,  0.6646 ,  1.39   , ...,  2.861  , -0.8853 ,...

