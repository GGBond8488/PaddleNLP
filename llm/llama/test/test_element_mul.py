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

def forward(hidden_states, weight):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = paddle.rsqrt(variance + variance_epsilon) * hidden_states

        if weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, weight.dtype)
        return hidden_states * weight



def test_cinn(x, w):
    emb = paddle.nn.functional.embedding(
        x=x, weight=w, name="embedding")
    tmp_6 = forward(emb, create_parameter)
    return tmp_6, emb



static_test = paddle.jit.to_static(test_cinn)


output,emb = static_test(ids, w)
paddle.save(output, './no_cinn.pdtensor')
paddle.save(emb, './no_cinn_emb.pdtensor')

# np.testing.assert_allclose(out_no_cinn, out_cinn, rtol=0)
# # np.testing.assert_allclose(out_no_cinn, tmp_6, rtol=1e-3)
# np.testing.assert_allclose(tmp_6, out_cinn, rtol=1e-3) 

# np.testing.assert_equal(out_no_cinn.numpy(), out_cinn.numpy())
# np.testing.assert_equal(out_no_cinn.numpy(), tmp_6.numpy())
# np.testing.assert_equal(tmp_6.numpy(), out_cinn.numpy())