---
title: 华为昇腾AscendC Vector类型算子Profile-Base性能优化实战
date: 2024-06-09 23:55:21
tags:
---

由于美国对中国高端芯片的封锁，为了避免英伟达GPU断供，所以需要支持华为昇腾的NPU。但是华为昇腾发展较晚，整体生态的完善度和研发工具的易用性都远远不及英伟达。本文从实战入手详细记录如何Profile-优化华为AscendC Vector算子。

# 一、 准备运行环境和代码
- 镜像https://ascendhub.huawei.com/#/detail/ascend-pytorch
- Ascend Toolkit 8.0RC2(相当于英伟达的CUDA)
- 910BC2(相当于英伟达的GPU型号：A100，H800等)
- 代码：一念https://github.com/pcg-mlp/KsanaLLM

# 二、编译运行华为算子

### 1. 编译代码
```bash
git clone https://git.woa.com/deep_learning_framework/KsanaLLM
cd KsanaLLM && mkdir build && cd build
cmake -DWITH_TESTING=ON \
      -DWITH_CUDA=OFF \
	  -DWITH_ACL=ON \
	  -DWITH_STANDALONE_TEST=ON ..
make -j
```
### 2. 运行
```bash
./bin/llm_kernels_ascend_permute_test
```
由于华为NPU核心计算部件分为AI Cube和AI Vector，所以本文实战案例分别以Vector计算为主的permute作为例子。

### 3. Profile Vector类型算子实例
采样运行数据
```bash
msprof op --application="./bin/llm_kernels_ascend_permute_test --gtest_filter=LlamaAscendPermuteTestSuit.PermuteKernelTest" \
          --aic-metrics=ArithmeticUtilization,L2Cache,Memory,MemoryL0,MemoryUB,PipeUtilization,ResourceConflictRatio \
	      --output=./output_data
```
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-1.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-1.png?raw=true)
导出分析报告
```bash
msprof --export=on --output=./output_data
```
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-2.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-2.png?raw=true)
调试命令中比较重要的是`--aic-metrics`。这个选项有ArithmeticUtilization，L2Cache，Memory，MemoryL0，MemoryUB，PipeUtilization，ResourceConflictRatio总共7个相关的指标分组。当导出分析报告之后，可以在运行采样目录./output_data下找到指标数据。
```
# 这个是op基础信息
./output_data/OPPROF_xx/OpBasicInfo.csv
./output_data/OPPROF_xx/PipeUtilization.csv
./output_data/OPPROF_xx/ArithmeticUtilization.csv
./output_data/OPPROF_xx/L2Cache.csv
./output_data/OPPROF_xx/Memory.csv
./output_data/OPPROF_xx/ResourceConflictRatio.csv
./output_data/OPPROF_xx/MemoryL0.csv
./output_data/OPPROF_xx/MemoryUB.csv
```
### 4. Vector类型算子Profile数据分析实战
首先我们先了解一下这个算子主要是用NPU上的哪个部件执行运算的。打开./output_data/OPPROF_xx/OpBasicInfo.csv如下:
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-3.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-3.png?raw=true)
华为昇腾NPU上的计算核心是AI Core，主要有2个计算单元：AI Cube和AI Vector。AI Cube主要处理矩阵计算任务，AI Vector主要处理向量运算任务。目前市面上华为NPU的AI Core有两种架构，AI Cube和AI Vector分离和统一架构。AI Cube和AI Vector分离是指两者不共享一个Unified Buffer，好处是两个计算单元可以独立并发执行。昇腾910B2C就是AI Cube和AI Vector分离架构。单个910B2C总共有24个AI Cube和48个AI Vector。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-4.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-4.png?raw=true)
上图是AI Cube/Vector统一架构
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-5.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-5.png?raw=true)
上图是AI Cube/Vector分离架构

因为我们目前在分析permute，permute执行Tensor维度交换的操作，所以OpType中显示的是Vector就是表示permute执行在AI Vector上。而Block dim表示用了多少个计算单元，图中显示1即表示使用了1个AI Vector。
#### 4.1 PipeUtilization
由于华为AscendC的计算范式如下图所示，是多级并发流水线的模式。所以PipeUtilization表示的是计算单元和搬运单元耗时占比。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-6.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-6.png?raw=true)
打开./output_data/OPPROF_xx/PipeUtilization.csv
发现aic_* 数据都是NA，这是因为不是矩阵运算，所以没有用到AIC（AI Cube）。这里可以着重分析AIV（AI Vector）。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-7.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-7.png?raw=true)
block_id表示逻辑意义上的AI Vector ID，sub_block_id表示物理意义上的AI Vector ID。例如NPU 910B2C上处理128个vector运算，后面统称为128个任务。block_id的值是0-47，sub_block_id的值是0-127，其中block_id为1的AI Vector要处理sub_block_id=1和sub_block_id=49的任务。
aiv_time(us)表示在这个AI Vector上执行128个任务中sub_block_id为0的任务的总耗时，单位是us。
aiv_total_cycles是执行这个任务时时钟总数。
aiv_vec_time(us)是代表vec类型指令（向量类运算指令）耗时。aiv_vec_ratio代表vec单元指令的cycle数在total cycle数中的占用比。
在AI Vector中，有Scalar单元和Vector处理单元。Scalar单元主要处理数据逻辑操作。例如:
```C++
// 下面语句都由scalar单元执行。
LocalTensor<T> a;
LocalTensor<T> b
__gm__ T* a_ptr = a[offset];
__gm__ T* b_ptr = b[offset];
// 下面这个语句由Vector单元执行。
vadd(a_ptr, b_ptr);
```
因为硬件上一般情况下计算部件单位时间处理数据量比传输通路单位时间传输数据量多（英伟达GPU和华为NPU在这一点上同理）。所以一般情况下aiv_time >= aiv_vec_time，且两者越接近越好，即aiv_vec_ratio越高越好。因为两者耗时越接近，逻辑控制流和数据搬运操作越少，越能发挥NPU的计算能力。
同理aiv_scalar_time(us)和aiv_scalar_ratio表示scalar单元总耗时和scalar类型指令（标量类运算指令）的clock cycle数在total clock cycle数中的占用比。因为scalar单元负责逻辑控制处理，所以这两个指标也是越低越好。
剩余的4个指标aiv_mte2_time(us)，aiv_mte2_ratio，aiv_mte3_time(us)，aiv_mte3_ratio均为数据搬运指标。MTE（Memory Transfer Engine）可以结合下面这个图来看
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-8.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-8.png?raw=true)
在Vector算子上，MTE2是Global Memory搬到Unified Memory的操作。MTE3是Unified Memory搬到Global Memory的操作。
aiv_icache_miss_rate：表示instruction cache缺失率，即未命中instruction的L1 cache，数值越小越好。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-9.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-9.png?raw=true)
通过分析PipeUtilization，发现整个计算过程中，scalar操作占比怀疑是因为scalar的占比过高导致的性能不足。
一般情况来说，只要分析到这里，然后用SIMD的vector指令代替scalar指令即可完成这个算子的初步优化，但是为了演示如何全面分析算子性能，后面会继续分析其他指标的profile文件。
#### 4.2 ArithmeticUtilization
ArithmeticUtilization同上，由于不是矩阵计算类型的算子，所以./output_data/OPPROF_xx/ArithmeticUtilization.csv中aic_为前缀的指标都显示为N/A，所以着重分析aiv指标。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-10.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-10.png?raw=true)
其中aiv_time，aiv_total_cycles，aiv_vec_ratio已经在前面章节描述过，所以这里可以忽略。
aiv_vec_<数据类型>\_ratio表示元素为<数据类型>的vector指令cycle数在total cycle数中的占用比。
aiv_vec\_fops表示vector类型浮点运算数，即计算量，可用于衡量算法/模型的复杂度，其中fops表示floating point operations，缩写为FLOPs。
可以看到，几乎完全没有使用AI Vector中vector组件进行计算，所以验证下一步最直接的优化方式是将scalar操作替换成vector的SIMD操作。
#### 4.3 AI Core的存储结构
英伟达GPU相似，华为NPU的存储结构是Global memory->L2->L1->L0->Unified Memory。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-11.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-11.png?raw=true)
 - GM（Global Memory）容量最大，通常是DDR或者HBM。GM为全部AI Core共享。
 - L2 Cache，L2缓冲区，在AI Core片上，介乎于GM和L1Cache中间，所有访存GM的操作都会被cache到L2上。
 - L1 Buffer，从L2读取数据作为缓存AI Cube的输入。AI Vector不使用此缓存。
 - L0 Buffer。一般L0 Buffer分两类，一类L0 buffer是AI Cube的输入，一类缓存AI Cube的输出，一般标记为L0C buffer。AI Vector不使用此缓存。
 - UB，Unified Buffer，用于缓存Vector和Scalar操作的输入和输出。通常对应逻辑中的LocalTensor，[huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-12.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-12.png?raw=true)
AI Cube不使用此缓存。
上述提及到的AI Cube/Vector统一/分离架构主要也是和存储结构相关。
与外部资料不一样的是，910B2C作为分离架构，AI Vector 和 AI Cube只存在GM共享关系。其他缓存均不共享，所以AI Cube的输出缓冲区的数据不能直接作为AI Vecor输入，必须先搬运到GM再搬运到UB。
##### 4.3.1 L2Cache
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-13.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-13.png?raw=true)
L2Cache中的指标一般和图中MTE2的性能相关。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-14.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-14.png?raw=true)
既然是与缓存相关，那么和访存指标一样，hit rate越高越好，miss rate越低越好。
##### 4.3.2 Memory
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-15.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-15.png?raw=true)
因为AI Vector只和GM-L2-UB相关，所以可以看到与L1 L0相关的指标全部都为N/A。Memory主要展现2个访存维度：UB和GM。
MTE2，MTE3和GM写入写出相关。
ub\*主要和AI Vector写入写出UB相关。
##### 4.3.4 MemoryUB
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-16.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-16.png?raw=true)
由于这个算子没有使用到AI Vector，只使用了scalar做数据搬运，所以这项也是空的。

#### 4.4 ResourceConflictRatio
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-17.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-17.png?raw=true)
这项指标和英伟达[CUDA bank conflict](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)的概念类似。Vector对UB的访存也是一个bank一个bank来的。所以也会存在bank的读写冲突或者bank group的读读冲突。这个文件就是展示这个冲突比率。bank conflict rate越低越好。
由于这个算子没有使用AI Vector。所以bank conflict rate全都是0。

### 5. Vector类型算子性能优化实战
Permute这个算子的原始实现https://github.com/pcg-mlp/KsanaLLM/blob/9201ca09f510244b3fa62b7360c0930508936995/3rdparty/LLM_kernels/csrc/kernels/ascend/permute/permute_kernel.cc:
```c++
template <typename T>
__aicore__ void PermuteKernel<T>::Process() {
  bool should_break = false;
  for (size_t i = 0; i < tiling_->dim0; ++i) {
    if (should_break) break;
    for (size_t j = 0; j < tiling_->dim1; ++j) {
      if (should_break) break;
      for (size_t k = 0; k < tiling_->dim2; ++k) {
        if (should_break) break;
        for (size_t x = 0; x < tiling_->dim3; ++x) {
          if (should_break) break;
          for (size_t y = 0; y < tiling_->dim4; ++y) {
            if (should_break) break;
            for (size_t z = 0; z < tiling_->dim5; ++z) {
              uint64_t src_pos = GetInputIndexPos(i, j, k, x, y, z);
              if (src_pos >= tiling_->block_length * (block_idx_ + 1) || src_pos >= tiling_->total_length) {
                should_break = true;
                break;
              }

              if (src_pos >= tiling_->block_length * block_idx_) {
                uint64_t dst_pos = GetNewIndexPos(i, j, k, x, y, z);
                *(const_cast<__gm__ T*>(output_gm_.GetPhyAddr()) + dst_pos) =
                    *(const_cast<__gm__ T*>(input_gm_.GetPhyAddr()) + src_pos);
              }
            }
          }
        }
      }
    }
  }
}
```
这个代码性能底下的根源是：
1. 用了六层循环套嵌来获取dims转换之间的坐标。
2. copy的时候也是scalar操作，即一个clock cycle只操作一个数。

所以最简单的一个优化方向是将最内层的copy转化成Vector的SIMD操作。

原始代码中最内层的z循环改成如下代码：
```c++
DataCopyParams simd_copy_param;
simd_copy_param.blockCount = 1;
simd_copy_param.blockLen = z;
simd_copy_param.srcStride = 1;
// 预先计算好新tensor在原z维度的相对于旧tensor的stride
simd_copy_param.dstStride = dst_z_stride;
DataCopy(tmpLocal, dstLocal, simd_copy_param);
```
这个代码有两个优化点:
 1. 输入输出从GM指针改成LocalTensor，因为LocalTensor存放在UB上。使用LocalTensor后便可以使用如下的UB<--->GM多级流水线提升性能。
![huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-18.png](https://github.com/whitelok/whitelok.github.com/blob/master/resources/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle/huawei-ascendc-vector-type-operator-profile-based-optimization-practical-battle-18.png?raw=true)
 2. 因为使用了Vector Copy指令，这样每一个clock cycle可以操作至少128个float16。这样算子的吞吐是原来的128倍。

### 6. 实验数据
最后我们将输入扩大至1024\*1024。端到端的时延从128.74ms降低到5ms。具体优化代码详见https://github.com/pcg-mlp/KsanaLLM/tree/9201ca09f510244b3fa62b7360c0930508936995 后的更新。
后续极致的性能优化将会同步到一念官方代码仓库中https://github.com/pcg-mlp/KsanaLLM。敬请期待。