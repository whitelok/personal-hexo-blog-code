---
title: 深度学习高性能异构框架TVM核心原理解释系列（3）TVM Halide入门 - 算法工程师如何闭眼写出高性能GPU计算代码
date: 2020-03-26 15:57:53
tags:
---



正如上一篇所说的，TVM借用了Halide的思想，将计算（compute）和用户计算加速的调度（schedule）分离，所以从业务场景上说，只要算法团队写好了算法，工程团队把schedule写好，就能用比较高效的方式部署业务应用了。



先用一个例子说明一下TVM是怎么做到计算和调度分离的



假如说用户想实现一个简单的矩阵按行reduction操作

```python
B = numpy.sum(A, axis=1) 
# 每行进行累加，结果放在第一个元素
# 若A=np.array([[1,2,3],[4,5,6],[7,8,9]])
# B: array([ 6, 15, 24])
```

那么TVM怎么把这个计算实现呢？

```python
B = topi.sum(A, axis=1)
```

对的这就实现了。

这时候有人就问，TVM操作靠谱不靠谱啊，talk is cheap, show me the code

```python
ts = te.create_schedule(B.op) # 创建调度实例
print(tvm.lower(ts, [A], simple_mode=True))
# 下面的输出结果就是TVM生成的伪代码
'''
allocate A_red[float32 * n]
produce A_red {
  for (ax0, 0, n) {
    A_red[ax0] = 0f
    for (k1, 0, m) {
      A_red[ax0] = (A_red[ax0] + A[((ax0*stride) + (k1*stride))])
    }
  }
}
'''
```

OK计算做好了，那么调度呢，并行优化呢，没看见你有写嘛

下面我们对这个简单的算法进行并行优化

优化思想如下：

![tvm-tutorials-lession-3-1.JPG](https://github.com/whitelok/whitelok.github.com/raw/master/resources/tvm-tutorials-lession-3-1.JPG?raw=true)

即多个thread并发执行，每个thread处理不同的行

为了让搞CUDA的同事失业，下面并行化在CUDA上实现

在保持当前算法不变的情况下，代码修改如下

```python
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import topi
import numpy as np

# 输入A，其中A为n×m的矩阵
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name='A')
# 定义计算，B的shape为长度为n的vector
B = topi.sum(A, axis=1)

#============一般来说算法同学完成上面的操作就可以了，
#============下面的由系统工程师来

# 创建调度实例
ts = te.create_schedule(B.op)

# 配置CUDA参数
# 用8个线程先试试水
num_thread = 8
# 设置用什么thread
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

# 只有一个dimension因为B为长度为n的vector
x_axis = ts[B].op.axis[0]

# 每个B中的元素都是由不同的thread生成的，所以我们可以让thread_x去处理操作得出每个
# B中的元素
ts[B].bind(x_axis, thread_x)

# 生成CUDA函数，并跑起来
func = tvm.build(ts, [A, B], 'cuda')

# 看看生成的kernel长啥样
'''
看着还行，和手写cuda还真差不多
// attr [A_red] storage_scope = "global"
allocate A_red[float32 * n]
produce A_red {
  // attr [iter_var(threadIdx.x, range(min=0, ext=8), threadIdx.x)] thread_extent = n
  A_red[threadIdx.x] = 0f
  for (k1, 0, m) {
    A_red[threadIdx.x] = (A_red[threadIdx.x] + A[((threadIdx.x*stride) + (k1*stride))])
  }
}
'''
# 跑起来看看
ctx = tvm.gpu(0)
n = 3
m = 3
a_np = np.array([[1,2,3],[4,5,6],[7,8,9]]).astype(A.dtype)

a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(np.zeros((n, ), dtype=B.dtype), ctx)
func(a, b)

# 用numpy算一下结果
b_np = numpy.sum(a_np, axis=1) 

# 看看结果能不能对得上，这里不报错就是通过了
tvm.testing.assert_allclose(b_np, b.asnumpy(), rtol=1e-2)

# 计算TVM运行时间
# 这里需要把n, m变大效果才明显
# 迭代400次消除噪声

evaluator = func.time_evaluator(func.entry_name, ctx, number=400)
```



#### 实验结果

*这里需要把n, m变大效果才明显*

CPU: Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz

|                | TVM CUDA GTX1080（ms） | numpy CPU 单核（ms） |
| -------------- | ---------------------- | -------------------- |
| n=1024, m=1024 | 0.684666               | 98.118               |
| n=1024, m=4096 | 2.627865               | 378.038883           |



当然大家完全可以基于这个框架部署不同的算法，有机会再讲



#### 后记

有算法并行加速问题欢迎找我讨论，love & peace