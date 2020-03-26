---
title: 深度学习高性能异构框架TVM核心原理解释系列（2）-TVM中循环计算自动并行化方法(以Loo.py为例)
date: 2019-06-25 16:01:58
tags:
---
搞过高性能计算的同学都知道，带有for循环的逻辑的串行代码，性能一般都不太好，特别是多重for循环套嵌的。如果对多重复杂循环的代码进行并行化，不仅需要对当前使用的计算环境的系统结构有很好的了解，而且还需要了解串行代码的业务逻辑，所以对串行代码优化是业务得到验证之后的通点和难点。

**那么有没有可能自动地将串行代码中的复杂循环块自动生成高效的并行化代码呢？**

通过多面体模型编译技术完全可以做到的，如GCC的Graphite框架、LLVM的Polly模块以及多面体模型在Open64和IBM XL编译器中的应用。

本文以TVM文档提及借鉴的Loo.py为例，通过实例分析，循环代码的自动并行化的实现。

首先介绍一下Loo.py，Loo.py是**多面体模型编译技术**的一个Python实现。

Loo.py的核心思想是通过对程序的依赖，边界的分析，将运行过程抽象成一个几何立体结果。再通过几何操作将 几何立体拆分成不同的部分分别重构计算过程，从而将这个计算转换成能够实现高性能的版本。

TVM中的NNVM模块通过引入多面体模型来解决计算图自动并行优化。



# Loo.py主要功能

 - OpenCL/CUDA 下的向量多核并行化
 - 数据在内存表示形式转换 (对象中的数组SoA转换成对象数组AoS)

  ```c
  struct {
      uint8_t r, g, b;
  } AoS[N];
  
  struct {
      uint8_t r[N];
      uint8_t g[N];
      uint8_t b[N];
  } SoA;
  ```

 - 循环展开
 - 通过处理边界问题将循环代码进行低维度的延展
 - 数据预加载和拷贝优化
 - 指令级并行优化
 - More…….



# 目前已经能证明被Loo.py优化的算法有哪些

 - 非稀疏向量/矩阵的线性代数运算
 - 卷积(对卷积也是有for循环的)
 - n-body 粒子模拟
 - 偏微分方程求解
 - 类似Resnet结构的网络模型backword和forward



# Loo.py的并行化原理

从计算机功能性的角度来说，多面体模型的应用框架loo.py可以说是传统编译过程的一个插件，他与传统的编译器又有一定的区别。传统编译器中，无论是否有用户注释的帮助，gcc/g++/clang之类的编译器都等价地将用户代码重写成机器指令让机器执行。但是loo.py是将用户的代码分析构造成多面体模型，然后进行并行化，输出kernel代码（CUDA并行化执行的单元），编译执行。

归结起来，loo.py的核心原理就是基于多面体模型的自动并行化技术。

**用人话说，就是将你的循环代码变成立体几何，然后通过立体几何变换成多个模块，再将这些模块变回代码，并行化执行这些代码**

![循环代码自动并行化简单事例图](https://github.com/whitelok/tvm-lesson/raw/master/lession-2/images/img-1.png)

多面体模型(Polyhedral Model)自动并行化是一种基于线性代数来表示程序和程序转换的计算模型，它应用了丰富的数学理论和直观的几何解释，且作为抽象语法树(AST)的改进，适合表示串行以及并行程序，并为分析和应用程序转换提供了方便的抽象模型。在程序的自动并行化和优化的处理方面上，已经通过应用多面体模型，已经取得了巨大的成效。Loo.py的实现中，数据模型正是借鉴了多面体模型的思想，以便对代码进行深度优化。

多面体模型自动并行化的主要步骤如下：
 1. 首先，从抽象语法树开始，将适合多面体模型的部分程序翻译成线性代数表示；
 2. 下一步，通过使用一种重新排序函数，来选择新的代码执行顺序。**如何来寻找最为适合的代码执行顺序正是大多于对于多面体模型研究的重点，并行化出来的代码快不快主要也是这部分决定的**；
 3. 最后，进行代码生成，返回原有抽象生成树，或实现根据代码重排函数指定的执行顺序的新的源代码；

本章所谈及的多面体模型涉及到大量的算法和公式，所以如果想对多面体模型技术有更深的了解请参考软件学报的 综述：[基于多面体模型的编译“黑魔法”](http://www.jos.org.cn/html/2018/8/5563.htm)



# Loo.py实例



### 简单的for循环

下面通过一个实例来演示一下如果将最简单的单层循环通过Loo.py自动化生成性能的并行代码：

首先先做一些初始化的工作

```python
import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
lp.set_caching_enabled(False)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
```

然后我们就可以开始写我们的循环体了，为了熟悉代码，我们先写一个简单的一重循环

```python
knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

# 上面的代码相当于Python版的一重循环
for i in range(n):
	out[i] = 2 * a[i]  
```

接着我们需要定义一下循环变量i，有人可能好奇地问，i不是已经定义了吗？`[i]: 0<=i<n` 。但是由于i的范围并不知道(n还没赋值)，而且在进行多面体构建的时候，需要指定一个类似于坐标轴一样的东西，所以需要对i进行更细化的明晰。而在Loo.py中，这种循环变量（每次循环都产生变化的变量）称为iname。

```python
knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
```

至此，我们的例子可以进入执行阶段了，而且这个阶段可以查看我们生成的并行化代码

```python
# 初始化一下输入变量
n = 128
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)

# 执行 
evt, (out,) = knl(queue, a=x_vec_dev)
# 输出结果
print(out)

# 添加类型
knl = lp.add_and_infer_dtypes(knl, {"a": np.dtype(np.float32)})
# 输出生成的并行化代码
print(lp.generate_code_v2(knl).device_code())
```

可以看到我们生成的并行化代码(OpenCL)是

```cpp
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(128, 1, 1))) loopy_kernel(__global float const *__restrict__ a, int const n, __global float *__restrict__ out)
{
  if (-1 + -128 * gid(0) + -1 * lid(0) + n >= 0)
    out[128 * gid(0) + lid(0)] = 2.0f * a[128 * gid(0) + lid(0)];
}
```

通过分析上面的代码，我们可以看到，原来的for不见了，取而代之的是kernel中的if语句。for语句中都逻辑并行已经通过多线程完成掉了（其实是逻辑先转换成几何模型，通过几何模型变换变成可以适合并行化的操作）。接下来我们看复杂一点的例子。



### 多重循环套嵌

```python
import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

knl = lp.make_kernel(
    "{[ictr,itgt,idim]: "
    "0<=itgt<ntargets "
    "and 0<=ictr<ncenters "
    "and 0<=idim<ambient_dim}",
"""
for itgt
    for ictr
        <> dist_sq = sum(idim,
                (tgt[idim,itgt] - center[idim,ictr])**2)
        <> in_disk = dist_sq < (radius[ictr]*1.05)**2
        <> matches = (
                (in_disk
                    and qbx_forced_limit == 0)
                or (in_disk
                        and qbx_forced_limit != 0
                        and qbx_forced_limit * center_side[ictr] > 0)
                )

        <> post_dist_sq = if(matches, dist_sq, HUGE)
    end
    <> min_dist_sq, <> min_ictr = argmin(ictr, ictr, post_dist_sq)

    tgt_to_qbx_center[itgt] = if(min_dist_sq < HUGE, min_ictr, -1)
end
""")

knl = lp.fix_parameters(knl, ambient_dim=2)
knl = lp.add_and_infer_dtypes(knl, {
        "tgt,center,radius,HUGE": np.float32,
        "center_side,qbx_forced_limit": np.int32,
        })
print(lp.generate_code_v2(knl).device_code())
```

生成的kenerl代码为

```c++
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
inline int loopy_argmin_int32_float32_op(
    int op1, float index1,
    int op2, float index2,
    float *index_out)
{
    if (op2 <= op1)
    {
        *index_out = index2;
        return op2;
    }
    else
    {
        *index_out = index1;
        return op1;
    }
}

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(float const HUGE, __global float const *__restrict__ center, __global int const *__restrict__ center_side, int const ncenters, int const ntargets, int const qbx_forced_limit, __global float const *__restrict__ radius, __global float const *__restrict__ tgt, __global float *__restrict__ tgt_to_qbx_center)
{
  int acc_ictr;
  float acc_ictr_0;
  float acc_idim;
  float dist_sq;
  int in_disk;
  int matches;
  int min_dist_sq;
  float min_ictr;
  float post_dist_sq;

  for (int itgt = 0; itgt <= -1 + ntargets; ++itgt)
    if (-1 + ncenters >= 0)
    {
      acc_ictr = INT_MAX;
      acc_ictr_0 = -1.0f;
      for (int ictr = 0; ictr <= -1 + ncenters; ++ictr)
      {
        acc_idim = 0.0f;
        for (int idim = 0; idim <= 1; ++idim)
          acc_idim = acc_idim + (tgt[ntargets * idim + itgt] + -1.0f * center[ncenters * idim + ictr]) * (tgt[ntargets * idim + itgt] + -1.0f * center[ncenters * idim + ictr]);
        dist_sq = acc_idim;
        in_disk = dist_sq < radius[ictr] * 1.05f * radius[ictr] * 1.05f;
        matches = (in_disk && qbx_forced_limit == 0) || (in_disk && qbx_forced_limit != 0 && qbx_forced_limit * center_side[ictr] > 0);
        post_dist_sq = (matches ? dist_sq : HUGE);
        acc_ictr = loopy_argmin_int32_float32_op(acc_ictr, acc_ictr_0, ictr, post_dist_sq, &(acc_ictr_0));
      }
      min_dist_sq = acc_ictr;
      min_ictr = acc_ictr_0;
      tgt_to_qbx_center[itgt] = (min_dist_sq < HUGE ? min_ictr : -1.0f);
    }
}
```



### 实验结果
通过实验，在RTX2080Ti上，多重循环的代码运行起来是原来for循环代码耗时的5分之一。

