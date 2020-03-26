---
title: 深度学习高性能异构框架TVM核心原理解释系列（1）-手把手教你用TVM做Inference加速
date: 2019-06-25 15:04:23
tags:
---
# TVM是什么？

官方说法：TVM是一种把Deep Learning（以下简称DL）模型分发到各种硬件设备上的、使得模型Inference性能达到最优的、端到端的解决方案。最终目标是让DL模型可以轻松部署到所有硬件种类中，其中不仅包括 GPU、FPGA 和 ASIC（如谷歌 TPU），甚至是嵌入式设备。

引用同在DMLC小组的刘洪亮（phunter_lau）进一步在微博上解释了这个工作的意义：“TVM可以把DL模型部署到不同硬件，比如群众常问的能不能用AMD的GPU，用FPGA怎么搞，TVM提供这个中间层有效解决这个问题”。

作者：陈天奇，华盛顿大学计算机系博士生，此前毕业于上海交通大学ACM班。XGBoost、cxxnet等著名机器学习工具的作者，MXNet的主要贡献者之一。

# 为什么要用TVM？

在Deep Learning（以下简称DL）模型的开发过程中一般分为2个阶段：Training和Inference，一般市面上大众框架例如：PyTorch，Tensorflow，Caffe等都包括这两个过程。然而，大部分DL框架的Inference过程并没有很好的优化，即使有如XLA之辈的优化，也只支持CPU，GPU等运算部件，并不能支持如NPU，FPGA等异构加速部件。

![TVM与其他框架之间的对比](https://github.com/whitelok/tvm-lesson/raw/master/lesson-1/images/img-1.png)

TVM相对于Tensorflow、Pytorch等DL框架有以下**优点**：

1. 部署简单，支持多种模型格式：NNVM，ONNX，Tensorflow (forzen pb), MXnet模型，DarkNet模型
2. 支持多种硬件平台及使用环境上的模型Inference：ARM(树莓派，iPhone)，FPGA，CPU，nVidia GPU，Mali GPU等。
3. 支持客户端-服务器的RPC调用。
4. 能最大地提高模型的Inference性能。
5. 不需要依赖部署庞大繁重DL框架才能进行Inference，更适合嵌入式场景。
6. 有强大的开源社区支持。

TVM相对于Tensorflow、Pytorch等DL框架有以下**缺点**：

1. 不能用于Training。
2. 不支持非开放接口的硬件，例如：TPU，华为的麒麟芯片等。
3. 需要手动支持不同框架下的新的op类型，例如：当将Pytorch模型转换成onnx模型后，可能会出现Aten op，但是这个TVM暂时没有实现，需要手动支持。

# 怎么使用TVM

TVM主要包括两个过程：编译和部署运行，如图所示。

![TVM主要过程](https://github.com/whitelok/tvm-lesson/raw/master/lesson-1/images/img-2.png)

  1. 图中NNVM表示层主要是将其他DL框架的模型的计算图加载成TVM自己实现的计算图表示层---NNVM。图表示层主要通过Halide，loo.py等框架做一些layer，op的融合与优化。（具体原理以及详细实例将在后续文章系列阐述。）以onnx模型为例，其代码结构大致如下：

```
onnx_model = onnx.load_model([模型地址])
sym, params = nnvm.frontend.from_onnx(onnx_model)
```

  2. NNVM计算图编译则将前一步的layer fusion等图优化策略加以实现，抽象成拓扑结构，保存或者直接进入下一步。

```
# 此处说明使用GPU作为inference的异构加速部件
target = 'cuda'
nnvm.compiler.build_config(opt_level=3)
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

# 图优化后的结果可以通过导出成不同的文件供不同编程语言的代码使用
lib.export_library("model/super_resolution.so")
with open('model/super_resolution.graph', 'w') as _f:
        _f.write(graph.json())
with open('model/super_resolution.params', 'wb') as _f:
        _f.write(nnvm.compiler.save_param_dict(params))
```

  3. 最后在不同的编程语言上导入图优化后的结果，进行inference即可。
```
# python版
# 导入模型
lib = tvm.module.load("model/super_resolution.so")
with open('model/super_resolution.graph', 'r') as _f:
        graph = nnvm.graph.load_json(_f.read())
with open('model/super_resolution.params', 'rb') as _f:
        params = nnvm.compiler.load_param_dict(_f.read())
# 创建部署运行环境
ctx = tvm.gpu(0)
dtype = 'float32' # 若是在ARM上模型量化成INT8将会有极大的性能提升。
m = graph_runtime.create(graph, lib, ctx)
```

```
# C++ 版
tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_dylib, device_type, device_id);
......
分配输入输出blob
......
tvm::runtime::PackedFunc run = mod.GetFunction("run");
// 执行inference
run();
```

# TVM的Inference性能测评实验结果

模型：[super_resolution.onnx](https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx)

实验源码：https://github.com/whitelok/tvm-lesson

测试机器：[GTX1080 with MAX-Q](https://detail.tmall.com/item.htm?spm=a230r.1.14.8.528b3f1dTYWhfP&id=567118762627&ns=1&abbucket=4&sku_properties=5919063:6536025)

| Inference框架                                | Inference 平均时间 | 显存占用 |
| -------------------------------------------- | ------------------ | -------- |
| Pytorch CUDA mode                            | 约2.86ms           | 623MB    |
| TVM Python CUDA mode                         | 约1.71ms           | 201MB    |
| TVM C++ CUDA mode                            | 约1.46ms           | 201MB    |

# P.S. 第一篇博客的最后

 - 关于原理、优缺点、特性，本人以后会有更深入的原理详述和源码导读。

 - 如果有同学能贡献个FPGA我做实验，我还是很愿意提供FPGA的实验结果的。

