import os

import paddle



def get_files(dir):
    return os.listdir(dir)

dir1 = "./save_cinn_2"
dir2 = "./save_comp_2"

cinn = get_files(dir1)
comp = get_files(dir2)

same_files = list(set(cinn).intersection(set(comp)))

count=0
same_files = ['tmp_18@GRAD_1']
for file in same_files:
    print("start to compare: ", file, )
    comp_tensor = paddle.Tensor(paddle.base.core.load_tensor_c(dir2 + "/" +file))
    cinn_tensor = paddle.Tensor(paddle.base.core.load_tensor_c(dir1 + "/" +file))
    print(comp_tensor, cinn_tensor)
    if paddle.allclose(comp_tensor, cinn_tensor, rtol=1e-3):
        print(file+ "is same")
        count+=1
print(count)


