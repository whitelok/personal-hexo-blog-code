---
title: Nvidia嵌入式系统系列使用CUDA的unified memory性能分析
date: 2020-03-23 11:55:52
tags:
---

# 这篇blog讲啥？

 - 在NVIDIA的嵌入式产品上，使用cudaMallocManaged接口管理显存和内存在系统整体性能上会有比较明显的收益，收益主要体现在GPU activity上，memcpy可被省掉了，但是在x86上这个收益不存在。

 - 在嵌入式设备上使用Unified Memory能在不影响kernel的执行效率下，更好地减少访存的时延。

 - 使用cudaMallocManaged调试代码比cudaMalloc->cudaMemcpy方式方便多了，显存上放了啥gdb print一下就能看到，妈妈再也不用担心我CPU-GPU数据调试问题了

# 为啥会有这么一个猜想？

 1.  Jetson Xavier是是NVIDIA于2018年底出品的的新一代人工智能用途的嵌入式边缘计算硬件。可以理解为一个带GPU的嵌入式硬件。

 2. Jetson Xavier上内存和显存都放在同一块RAM上，理论上，东西放内存or显存，都在一个芯片上，是不是可以实现Zero Copy、而不是显式copy一下，而且嵌入式的带宽真是没法儿和x86比，memcpy一下真是慢出翔。

 3. 很多时候，调试GPU代码得看kernel输入输出的数据是不是对的，而gdb print不能直接print 显存数据，cuda-gdb跑起来简直慢到让人想摸鱼，所以有没一个更快的方法解决这个问题。

# Unified Memory

unified memory：在CUDA 6.0版本后，CUDA新增了统一寻址这个功能，该功能主要是在内存池结构上增加了一个统一内存系统，程序员可以直接访问任何内存/显存资源，或者在合法的内存空间内寻址，而不用管涉及到的存储空间到底是内存还是显存。

而涉及到内存和显存之间的数据拷贝由程序员的手动执行（cudaMemcpy），变成自动执行（yes，啥都不用干）。

# 实验

 - 设备：Jetson Xavier 
 - 系统：Jetpack 4.2
 - 显存占用：
  - 无Unified Memory: RAM 9292/15699MB
  - 有Unified Memory: RAM 9285/15699MB
 - 实验用kernel: 矩阵大小1000*1000 Byte的乘加（无share memory 无auto boost）
 - 统计对象GPU Activities:

有Unified Memory，kernel内循环100次


| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 100.00% | 22.2794s | 1     | 22.2794s | 22.2794s | 22.2794s | AplusB(int, int, int, int) |

无Unified Memory，kernel内循环100次


| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 93.86%  | 22.2528s | 1     | 22.2528s | 22.2528s | 22.2528s | AplusB(int, int, int, int) |
| 3.08%   | 730.93ms | 2     | 365.47ms | 360.19ms | 370.74ms | [CUDA memcpy HtoD]         |
| 3.06%   | 725.64ms | 1     | 725.64ms | 725.64ms | 725.64ms | [CUDA memcpy DtoH]         |

有Unified Memory，kernel内循环10次	

| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 100.00% | 2.29094s | 1     | 2.29094s | 2.29094s | 2.29094s | AplusB(int, int, int, int) |

无Unified Memory，kernel内循环10次

| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 60.99%  | 2.29096s | 1     | 2.29096s | 2.29096s | 2.29096s | AplusB(int, int, int, int) |
| 19.63%  | 737.42ms | 2     | 368.71ms | 364.59ms | 372.83ms | [CUDA memcpy HtoD]         |
| 19.37%  | 727.62ms | 1     | 727.62ms | 727.62ms | 727.62ms | [CUDA memcpy DtoH]         |

有Unified Memory，kernel外循环100次

| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 100.00% | 26.8297s | 100   | 268.30ms | 268.27ms | 268.32ms | AplusB(int, int, int, int) |

无Unified Memory，kernel外循环100次

| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 94.79%  | 26.8299s | 100   | 268.30ms | 268.27ms | 268.34ms | AplusB(int, int, int, int) |
| 2.65%   | 749.90ms | 2     | 374.95ms | 367.92ms | 381.97ms | [CUDA memcpy HtoD]         |
| 2.56%   | 725.41ms | 1     | 725.41ms | 725.41ms | 725.41ms | [CUDA memcpy DtoH]         |

有Unified Memory，kernel外循环10次	

| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 100.00% | 2.68303s | 10    | 268.30ms | 268.29ms | 268.33ms | AplusB(int, int, int, int) |

无UM，kernel外循环10次


| Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                       |
| ------- | -------- | ----- | -------- | -------- | -------- | -------------------------- |
| 64.62%  | 2.68300s | 10    | 268.30ms | 268.28ms | 268.31ms | AplusB(int, int, int, int) |
| 17.93%  | 744.61ms | 2     | 372.30ms | 366.02ms | 378.58ms | [CUDA memcpy HtoD]         |
| 17.44%  | 724.20ms | 1     | 724.20ms | 724.20ms | 724.20ms | [CUDA memcpy DtoH]         |

# 结论

在嵌入式设备上使用Unified Memory确实能在不影响kernel的执行效率下，更好地减少访存的时延，当然，本实验主要是Matrix Multiply kernel，如果可以，可以试试不同的算法，例如传统的高访存machine learning算法。