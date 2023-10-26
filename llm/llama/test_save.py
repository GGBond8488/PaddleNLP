import paddle
paddle.set_flags({'FLAGS_save_load_path': './save/'})
paddle.set_flags({'FLAGS_save_tensor': True})

@paddle.jit.to_static
def net(x):
    a = paddle.scale(x, 2)
    b = paddle.add(a, a)
    d = paddle.sum(b)
    paddle.grad([d], [x])
    return d


input = paddle.ones([3,4]).cast('float16')
input.stop_gradient=False

out = net(input)
print(out)
out.backward()

print(input.grad)

# a = paddle.base.core.load_tensor_c('./save/cinn_launch-out-_jst.0.x.0@GRAD_1')
# print(paddle.Tensor(a))