---
title: Tensorflow 1.8 计算图膨胀问题
date: 2018-05-14 15:04:23
tags:
---

在使用Tensorflow的过程中，经常在训练循环中使用如下的写法

```python
lr = tf.Variable(0.0,trainable=False)
......
with tf.Session() as sess:
    ......
    for _ in range(10):
        sess.run(tf.assign(lr, lr_rate*0.9**epoch))
    ......
```

在for过程中通过``print len(tf.get_default_graph().get_operations())``发现graph越来越大，而且执行速度越来越慢。

这是因为在sess.run的过程中，由于运算没有赋值给Variable，所以每次run的时候都在graph中生成新的node。
新建的node需要重新加载数据，所以如果使用cpu数据需要在内存新分配一份，如果使用gpu那么需要新声明一块显存地址并且将数据从内存上拷贝过去。

解决方法是：在函数调用中尽量使用变量代替运算式。例如上，上面的代码可以改写成这样

```python
lr = tf.Variable(0.0,trainable=False)
param = lr_rate*0.9**epoch
......
with tf.Session() as sess:
    ......
    for _ in range(10):
        sess.run(tf.assign(lr, param))
    ......
```
