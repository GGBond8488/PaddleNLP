from turtle import position
import paddle
import numpy as np

paddle.set_device('gpu:0')

paddle.set_default_dtype("float32")

embedding_0_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/lookup_table_v2-output-embedding_0.tmp_0_1"
embedding_0 = paddle.Tensor(paddle.base.core.load_tensor_c(embedding_0_path))

rsqrt_0_tmp_0_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_mul-input-rsqrt_0.tmp_0_1"
rsqrt_0 = paddle.randn([1, 2048, 4096], dtype="float32")

tmp_3 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/scale-output-tmp_3_1"

create_parameter_0_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/elementwise_mul-input-create_parameter_0.w_0_1"
create_parameter = paddle.Tensor(paddle.base.core.load_tensor_c(create_parameter_0_path))

ids = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/lookup_table_v2-input-_jst.0.input_ids.0_1"
ids = paddle.Tensor(paddle.base.core.load_tensor_c(ids))

w = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/lookup_table_v2-input-embedding_0.w_0_1"
w= paddle.Tensor(paddle.base.core.load_tensor_c(w))

variance_epsilon=1e-6

# def forward(hidden_states, weight):
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = paddle.rsqrt(variance + variance_epsilon) * hidden_states

#         if weight.dtype in [paddle.float16, paddle.bfloat16]:
#             hidden_states = paddle.cast(hidden_states, weight.dtype)
#         return hidden_states * weight



# def test_cinn(x, w):
#     emb = paddle.nn.functional.embedding(
#         x=x, weight=w, name="embedding")
#     tmp_6 = forward(emb, create_parameter)
#     return tmp_6, emb

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
    sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def test_cinn(q, k, cos, sin, position_ids):
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    return q_embed, k_embed


sin_path =  "/root/paddlejob/PaddleNLP/llm/llama/save_comp/cast-output-cast_4.tmp_0_1"
cos_path =  "/root/paddlejob/PaddleNLP/llm/llama/save_comp/cast-output-cast_5.tmp_0_1"
position_ids_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/expand_v2-output-expand_0.tmp_0_1"
q_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-output-reshape2_0.tmp_0_1"
k_path = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/reshape2-output-reshape2_0.tmp_0_1"

sin = paddle.Tensor(paddle.base.core.load_tensor_c(sin_path))
cos = paddle.Tensor(paddle.base.core.load_tensor_c(cos_path))
position_ids = paddle.Tensor(paddle.base.core.load_tensor_c(position_ids_path))
q = paddle.Tensor(paddle.base.core.load_tensor_c(q_path))
k = paddle.Tensor(paddle.base.core.load_tensor_c(k_path))

print(sin)
print(cos)
print(position_ids)
print(q)
print(k)

static_test = paddle.jit.to_static(test_cinn)


q_embed, k_embed = static_test(q, k, cos, sin, position_ids)
# paddle.save(q_embed, './cinn_fusion_q_embed.pdtensor')
# paddle.save(k_embed, './cinn_fusion_k_embed.pdtensor')
