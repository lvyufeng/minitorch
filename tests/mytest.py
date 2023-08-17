import minitorch

import minitorch.tensor as tensor

"""
#a = tensor([[1,2],[1,2]])
#b = tensor([[3,3],[4,4]])
a = tensor([1,2])
b = tensor([[3],[4]])
c = a * b
print(c)
c = c.sum()
print(c)
c.backward()

order = [2,0,1]
un_permute = [0 for _ in range(len(order))]
print(un_permute)

for d1, d2 in enumerate(order):
    un_permute[d2] = d1
print(un_permute)
"""


#TensorBackend = minitorch.make_tensor_backend(minitorch.TensorOps)
FastTensorBackend = minitorch.make_tensor_backend(minitorch.FastOps)
"""a = tensor([[1,2],[1,2]], backend = FastTensorBackend)
b = tensor([[3,3],[4,4]], backend = FastTensorBackend)
c = a@b
print(c)
"""

"""
import random
dim_c = 32
x = [[random.random() for i in range(dim_c)] for j in range(dim_c)]
y = [[random.random() for i in range(dim_c)] for j in range(dim_c)]

CudaTensorBackend = minitorch.make_tensor_backend(minitorch.CudaOps)
a = tensor(x, backend = CudaTensorBackend)
b = tensor(y, backend = CudaTensorBackend)
c = minitorch.mm_practice(a, b)

z = minitorch.tensor(x, backend=FastTensorBackend) @ minitorch.tensor(y, backend=FastTensorBackend)

for i in range(dim_c):
    for j in range(dim_c):
        if minitorch.operators.is_close(z[i, j], c._storage[dim_c * i + j]):
            pass
        else:
            print("("+str(i) + "," + str(j)+") "+ str(c._storage[dim_c * i + j]) + ", "+ str(z[i, j]))
"""

"""
import random
dim_c = 6
x1 = [[random.random() for i in range(dim_c)] for j in range(dim_c)]
x2 = [[random.random() for i in range(dim_c)] for j in range(dim_c)]
x = []
x.append(x1)
x.append(x2)
y = [[random.random() for i in range(dim_c)] for j in range(dim_c)]

CudaTensorBackend = minitorch.make_tensor_backend(minitorch.CudaOps)
a = tensor(x, backend = CudaTensorBackend)
b = tensor(y, backend = CudaTensorBackend)
c = a@b

z = minitorch.tensor(x, backend=FastTensorBackend) @ minitorch.tensor(y, backend=FastTensorBackend)

for i in range(dim_c):
    for j in range(dim_c):
        if minitorch.operators.is_close(z[0, i, j], c._tensor._storage[dim_c * i + j]):
            pass
        else:
            print("("+str(i) + "," + str(j)+") "+ str(c._tensor._storage[dim_c * i + j]) + ", "+ str(z[0, i, j]))
"""

"""
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
mndata = MNIST("C:/SciData/DeepleranData/MNIST/raw")
images, labels = mndata.load_training()
im = np.array(images[0])
im = im.reshape(28, 28)
plt.imshow(im)
plt.show()
"""



t = minitorch.rand(shape=(1, 1, 4, 4), backend = minitorch.TensorFunctions)
"""
out = minitorch.avgpool2d(t, (2, 2))
assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
)
out = minitorch.avgpool2d(t, (2, 1))
assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
)
"""

"""
out = minitorch.avgpool2d(t, (1, 2))
assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
)


fnlout = out.sum()
print(fnlout)
print(fnlout.shape)

print("backward")
fnlout.backward()
"""
#minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


input = minitorch.rand(shape=(1, 1, 4, 4), backend = minitorch.TensorFunctions)
batch, channel, height, width = input.shape
kh, kw = (1,2)
tile_h = height // kh
tile_w = width // kw

print("** 0 **")
print(input)
print(input.shape)
print()

input = input.contiguous()
input = input.view(batch, channel, tile_h, kh, tile_w, kw)
print("** 1 **")
print(input)
print("input = input.view(batch, channel, tile_h, kh, tile_w, kw) ")
print(input.shape)
print()

input = input.permute(0, 1, 2, 4, 3, 5)
print("** 2 **")
print(input)
print(input.shape)
print()

input = input.contiguous()
input = input.view(batch, channel, tile_h, tile_w, kw * kh)
print("** 3 **")
print(input)
print(input.shape)
print()

input = input.mean(4)
print("** 4 **")
print(input)
print(input.shape)
print()

input = input.view(batch, channel, tile_h, tile_w)
print("** 5 **")
print(input)
print(input.shape)
print()

input = input.sum()
print("** 6 **")
print(input)
print(input.shape)
print()

print("start backward")
input.backward()

