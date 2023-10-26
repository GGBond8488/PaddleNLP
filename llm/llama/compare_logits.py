import numpy as np
import paddle



comp = paddle.load('no_cinn_output.paddle')

cinn = paddle.load('cinn_output.paddle')


print(comp, cinn)

np.testing.assert_allclose(comp, cinn, atol=1e-3, rtol=1e-3)
