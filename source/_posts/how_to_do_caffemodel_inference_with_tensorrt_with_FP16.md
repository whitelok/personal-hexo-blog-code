---
title: 如何在TensorRT上用半精度(FP16)对Caffemodel进行inference
date: 2018-09-05 15:04:23
tags:
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

1. Why? 为什么我们需要使用FP16半精度进行Inference
	1. 低的数据精度代表占用更少的存储资源，例如：FP32即float占用的是4字节，FP16占用的2字节，FP16存储空间占用率只有float的一半
	2. 计算单元处理FP16比处理FP32的速度快
	3. Nvidia有大量的计算加速框架(cuDNN, cuBlas)以及加速硬件(Tensor Core, DLA)对FP16的数据有加速优化

2. NVidia是如何处理float与FP16的转换的，和其他普通的框架处理精度问题之间的区别是什么?
	在IEEE-754的描述中单精度浮点数(Nvidia习惯称为FP32，C++标准中的float)是4个字节，包括1位符号、8位指数和23位尾数。\\(value = sign × exponent × fraction\\) 浮点数的实际值\\(value\\)，等于符号位（sign bit）乘以指数偏移值(exponent bias)再乘以分数值(fraction)。
	NVidia在2002年提出了半精度浮点数FP16，只使用2个字节16位，包括1位符号、5位指数和10位尾数，动态范围是 \\( 2^{-30}\sim 2^{31} \\)也就是 \\(10^{-9}\sim 10^9 \\)，精度是 \\( \rm lg2^{11} \\)，大约3个十进制有效数字。NVidia的方案已经被IEEE-754采纳。Google的TensorFlow则比较简单粗暴，把单精度的后16位砍掉，也就是1位符号、8位指数和7位尾数。动态范围和单精度相同，精度只有 \\( \rm lg 2^8 \\)，2个有效数字。
	1. NVidia的float与FP16的转换方法:
	***float --> FP16***
	```
	typedef unsigned short half;
	half nvFloat2Half(float m)
	{
	    unsigned long m2 = *(unsigned long*)(&m);    
	    // 强制把float转为unsigned long
	    // 截取后23位尾数，右移13位，剩余10位；符号位直接右移16位；
	    // 指数位麻烦一些，截取指数的8位先右移13位(左边多出3位不管了)
	    // 之前是0~255表示-127~128, 调整之后变成0~31表示-15~16
	    // 因此要减去127-15=112(在左移10位的位置).
	    unsigned short t = ((m2 & 0x007fffff) >> 13) | ((m2 & 0x80000000) >> 16) 
	        | (((m2 & 0x7f800000) >> 13) - (112 << 10));           
	    if(m2 & 0x1000) 
	        t++;                       // 四舍五入(尾数被截掉部分的最高位为1, 则尾数剩余部分+1)
	    half h = *(half*)(&t);     // 强制转为half
	    return h ;
	}
	```
	***FP16 --> float***
	```
	float nvHalf2Float(half n)
	{
	    unsigned short frac = (n & 0x3ff) | 0x400;
	    int exp = ((n & 0x7c00) >> 10) - 25;
	    float m;
	    if(frac == 0 && exp == 0x1f)
	        m = INFINITY;
	    else if (frac || exp)
	        m = frac * pow(2, exp);
	    else
	        m = 0;
	    return (n & 0x8000) ? -m : m;
	}
	```
	2. 其他框架的典型如Tensorflow的float与FP16的转换方法:
	***float <--> FP16***
	```
	class Float16Compressor
	{
	    union Bits
	    {
	        float f;
	        int32_t si;
	        uint32_t ui;
	    };
	    static int const shift = 13;
	    static int const shiftSign = 16;
	    static int32_t const infN = 0x7F800000;//flt32 infinity
	    static int32_t const maxN = 0x477FE000;//max flt16 normal as a flt32
	    static int32_t const minN = 0x38800000;//min flt16 normal as a flt32
	    static int32_t const signN = 0x80000000;//flt32 sign bit
	    static int32_t const infC = infN>> shift;
	    static int32_t const nanN = (infC + 1) <<shift;//minimum flt16 nan as a flt32
	    static int32_t const maxC = maxN>> shift;
	    static int32_t const minC = minN>> shift;
	    static int32_t const signC = signN>> shiftSign;//flt16 sign bit
	    static int32_t const mulN = 0x52000000;//(1 <<23)/minN
	    static int32_t const mulC = 0x33800000;//minN/(1 <<(23 - shift))
	    static int32_t const subC = 0x003FF;//max flt32 subnormal down shifted
	    static int32_t const norC = 0x00400;//min flt32 normal down shifted
	    static int32_t const maxD = infC - maxC - 1;
	    static int32_t const minD = minC - subC - 1;
	public:
	    static float decompress(uint16_t value)
	    {
	        Bits v;
	        v.ui = value;
	        int32_t sign = v.si & signC;
	        v.si ^= sign;
	        sign <<= shiftSign;
	        v.si ^= ((v.si + minD) ^ v.si) & -(v.si> subC);
	        v.si ^= ((v.si + maxD) ^ v.si) & -(v.si> maxC);
	        Bits s;
	        s.si = mulC;
	        s.f *= v.si;
	        int32_t mask = -(norC> v.si);
	        v.si <<= shift;
	        v.si ^= (s.si ^ v.si) & mask;
	        v.si |= sign;
	        return v.f;
	    }
	    static uint16_t compress(float value)
	    {
	        Bits v, s;
	        v.f = value;
	        uint32_t sign = v.si & signN;
	        v.si ^= sign;
	        sign>>= shiftSign; //logical shift
	        s.si = mulN;
	        s.si = s.f * v.f; //correct subnormals
	        v.si ^= (s.si ^ v.si) & -(minN> v.si);
	        v.si ^= (infN ^ v.si) & -((infN> v.si) & (v.si> maxN));
	        v.si ^= (nanN ^ v.si) & -((nanN> v.si) & (v.si> infN));
	        v.ui>>= shift; //logical shift
	        v.si ^= ((v.si - maxD) ^ v.si) & -(v.si> maxC);
	        v.si ^= ((v.si - minD) ^ v.si) & -(v.si> subC);
	        return v.ui | sign;
	    }
	};
	```

3. How? 当我们手上有一个caffemodel的时候，怎么才能使用到TensorRT的FP16 Inference能力呢？
	1). 下载[`nvcaffe`](https://github.com/whitelok/nvcaffe-for-fp16-feature)并编译`cmake -DTEST_FP16=ON`
	2). `export PYTHONPATH=/the/path/to/your/caffe_root/python/`
	3). 修改模型文件prototxt:
		a. 添加`default_forward_type:  FLOAT16`到prototxt头部
		b. 添加`default_backward_type: FLOAT16`到prototxt头部
	4). 转换FP32模型成FP16兼容模式文件`python genDefaultModel.py --input /path/to/net.prototxt --model /path/to/net_fp32.caffemodel --output /path/to/net_fp16.caffemodel`

