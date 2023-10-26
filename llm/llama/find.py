import numpy as np
import paddle





# Start run Place(gpu:0) Op(rsqrt_grad), 
# inputs:{Out[rsqrt_0.tmp_0:::paddle::platform::float16[1, 2048, 1]({})(Place(gpu:0))], 
#         Out@GRAD[rsqrt_0.tmp_0@GRAD:::paddle::platform::float16[1, 2048, 1]({})(Place(gpu:0))]}, 
# outputs:{X@GRAD[tmp_20@GRAD:::paddle::platform::float16[]({})(Place(gpu:0))]}.



# b = paddle.base.core.load_tensor_c('./save2/cast-out-0_910')
# print(paddle.Tensor(b))

# rsqrt dout
# ./save2/elementwise_mul_grad-out-0_290

# rsqrt out
# ./save2/rsqrt-out-0_1

#  layer 1
# print("===============")
# print("layer 1")
# a = paddle.base.core.load_tensor_c('./save/rsqrt_grad-out-0_65')

# dout =  paddle.Tensor(paddle.base.core.load_tensor_c('./save/elementwise_mul_grad-out-0_290'))
# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_1'))
# print(dout)
# print("rsqrt out")



# # layer 9 
# # ./save/rsqrt-out-0_10
# # ./save/elementwise_mul_grad-out-0_248
# # ./save/rsqrt_grad-out-0_56
# print("===============")
# print("layer 9")
# a = paddle.base.core.load_tensor_c('./save/rsqrt_grad-out-0_56')

# dout =  paddle.Tensor(paddle.base.core.load_tensor_c('./save/elementwise_mul_grad-out-0_248'))
# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_10'))
# print(dout)


# # layer 22 
# # ./save/rsqrt-out-0_22
# # ./save/elementwise_mul_grad-out-0_194
# # ./save/rsqrt_grad-out-0_44
# print("===============")
# print("layer 22")
# a = paddle.base.core.load_tensor_c('./save/rsqrt_grad-out-0_44')
# dout =  paddle.Tensor(paddle.base.core.load_tensor_c('./save/elementwise_mul_grad-out-0_194'))
# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_22'))
# print(dout)


# # layer 32
# # ./save/rsqrt-out-0_33
# # ./save/elementwise_mul_grad-out-0_146
# # ./save/rsqrt_grad-out-0_33
# print("===============")
# print("layer 32")
# a = paddle.base.core.load_tensor_c('./save/rsqrt_grad-out-0_33')

# dout =  paddle.Tensor(paddle.base.core.load_tensor_c('./save/elementwise_mul_grad-out-0_146'))
# out = paddle.Tensor(paddle.base.core.load_tensor_c('./save/rsqrt-out-0_33'))
# print(dout)






# dx = -0.5 * dout * out * out * out

# np.savetxt( "dout.csv", dout.numpy().reshape(2048), fmt='%f', delimiter=",")
# np.savetxt( "out.csv", out.numpy().reshape(2048), fmt='%f', delimiter=",")
# print(dout.numpy())

# reduce_mean = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/reduce_mean-out-0_1'))
# print("reduce_mean 1 out")
# print(reduce_mean)

# scale_out = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/scale-out-0_2'))
# print("scale_out")
# print(scale_out)

# reduce_mean_2 = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/reduce_mean-out-0_2'))
# print("reduce mean 2 out")
# print(reduce_mean_2)

# #finally new path : ./save2/cast-out-0_8
 
# pow_1_input = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/cast-out-0_8'))
# print("pow 1 input")
# print(pow_1_input)

# # ./save2/cast-out-0_18

# pow_2_input = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/cast-out-0_18'))
# print("pow 2 input")
# print(pow_2_input)


# rs_out_2 = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/rsqrt-out-0_2'))
# print(rs_out_2)


# element_out_grad = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/elementwise_mul_grad-out-0_20'))
# print("element_out_grad")
# print(element_out_grad)

# rsqrt_1_out_grad = paddle.Tensor(paddle.base.core.load_tensor_c('./save2/rsqrt_grad-out-0_5'))
# print("rsqrt_1_out_grad")
# print(rsqrt_1_out_grad)

# sss = -0.5 * element_out_grad * out * out * out
# print(sss)

tmp_3 = "/root/paddlejob/PaddleNLP/llm/llama/save_comp/scale-output-tmp_3_1"
tmp_3 = paddle.Tensor(paddle.base.core.load_tensor_c(tmp_3))
print(tmp_3)