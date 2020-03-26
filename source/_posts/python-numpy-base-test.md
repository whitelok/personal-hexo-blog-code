---
title: Python NumPy笔试测试70题
date: 2018-04-13 15:04:23
tags:
---

1. 将NumPy导入，命名为别名np，并查看其版本
	答案：
	```python
	import numpy as np
	print np.__version__
	```

2. 如何创建1维数组？(创建数字从0到9的1维数组。)
	期望输出：
	```bash
	#> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	```
	答案：
	```python
	np.array(range(0,10))
	```

3. 如何创建boolean数组？创建所有元素均为True的3×3 NumPy数组。
	答案：
	```python
	np.full((3,3), True)
	```

4. 如何从1维数组中提取满足给定条件的项？（从 arr 中提取所有奇数元素。）
	输入：
	```python
	arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	```
	期望输出：
	```bash
	#> array([1, 3, 5, 7, 9])
	```
	答案：
	```python
	np.where(arr % 2 == 1)
	```

5. 如何将 NumPy 数组中满足给定条件的项替换成另一个数值？
	输入：
	```python
	arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	```
	期望输出：
	```bash
	#> array([ 0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
	```
	答案：
	```python
	arr[arr % 2 == 1] = -1
	```

6. 如何在不影响原始数组的前提下替换满足给定条件的项？
	输入：
	```python
	arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	期望输出：
	```bash
	#> array([ 0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
	```
	答案：
	```python
	out = arr.copy()
	out[out % 2 == 1] = -1
	```

7. 如何重塑（reshape）数组？(将1维数组转换成2维数组（两行）)
	输入：
	```python
	arr = np.arange(10)
	```
	期望输出：
	```bash
	#> array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
	```
	答案：
	```python
	np.reshape(arr, (2,-1))
	```

8. 如何垂直堆叠两个数组？(垂直堆叠数组a和b)
	输入：
	```python
	a = np.arange(10).reshape(2,-1)
	b = np.repeat(1, 10).reshape(2,-1)
	```
	期望输出：
	```bash
	#> array([[0, 1, 2, 3, 4],
	#> [5, 6, 7, 8, 9],
	#> [1, 1, 1, 1, 1],
	#> [1, 1, 1, 1, 1]])
	```
	答案：
	```python
	np.concatenate((a, b))
	```

9. 如何水平堆叠两个数组？
	输入：
	```python
	a = np.arange(10).reshape(2,-1)
	b = np.repeat(1, 10).reshape(2,-1)
	```
	期望输出：
	```bash
	#> array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
	#> [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
	```
	答案：
	```python
	np.concatenate((a, b), axis=1)
	```

10. 在不使用硬编码的前提下，如何在 NumPy中生成自定义序列？(在不使用硬编码的前提下创建以下模式。仅使用NumPy函数和以下输入数组 a)
	输入：
	```python
	a = np.array([1,2,3])
	```
	期望输出：
	```bash
	#> array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
	```
	答案：
	```python
	np.concatenate((a.repeat(3), np.tile(a,3)))
	```

11. 如何获得两个 Python NumPy 数组中共同的项？(获取数组 a 和 b 中的共同项。)
	输入：
	```python
	a = np.array([1,2,3,2,3,4,3,4,5,6])
	b = np.array([7,2,10,2,7,4,9,4,9,8])
	```
	期望输出：
	```bash
	#> array([2, 4])
	```
	答案：
	```python
	np.unique(a[a==b])
	```

12. 如何从一个数组中移除与另一个数组重复的项？
	输入：
	```python
	a = np.array([1,2,3,4,5])
	b = np.array([5,6,7,8,9])
	```
	期望输出：
	```bash
	array([1,2,3,4])
	```
	答案：
	```python
	np.array([x for x in a if x not in b])
	```

13. 如何获取两个数组匹配元素的位置？
	输入：
	```python
	a = np.array([1,2,3,2,3,4,3,4,5,6])
	b = np.array([7,2,10,2,7,4,9,4,9,8])
	```
	期望输出：
	```bash
	#> (array([1, 3, 5, 7]),)
	```
	答案：
	```python
	np.where(a==b)
	```

14. 如何从NumPy数组中提取给定范围内的所有数字？(从数组a中提取5和10之间的所有项）。
	输入：
	```python
	a = np.arange(15)
	```
	期望输出：
	```bash
	(array([ 5, 6, 7, 8, 9, 10]),)
	```
	答案：
	```python
	np.array([i for i in a if i >= 5 and i<=10])
	```

15. 如何创建一个Python函数以对NumPy数组执行元素级的操作？（转换函数maxx，使其从只能对比标量而变为对比两个数组。）
	输入：
	```python
	def maxx(x, y):
    if x >= y:
      return x
    else:
      return y
	maxx(1, 5)
	#> 5
	a = np.array([5, 7, 9, 8, 6, 4, 5])
	b = np.array([6, 3, 4, 8, 9, 7, 1])
	pair_max(a, b)
	```
	期望输出：
	```bash
	#> array([ 6., 7., 9., 8., 9., 7., 5.])
	```
	答案：
	```python
	pair_max = np.frompyfunc(maxx,2,1)
	```

16. 如何在2d NumPy数组中交换两个列？（在数组arr中交换列1和列2）。
	输入：
	```python
	arr = np.arange(9).reshape(3,3)
	```
	答案：
	```python
	arr[:,[0,1]] = arr[:,[1,0]]
	```

17. 如何在2d NumPy数组中交换两个行？（在数组arr中交换行1和行2）。
	输入：
	```python
	arr = np.arange(9).reshape(3,3)
	```
	答案：
	```python
	arr[[0,1], :] = arr[[1,0], :]
	```

18. 如何反转2D数组的所有行？（反转2D数组arr中的所有行）。
	输入：
	```python
	arr = np.arange(9).reshape(3,3)
	```
	答案：
	```python
	arr[:,:] = arr[: :-1,:]
	```

19. 如何反转2D数组的所有列？（反转2D数组arr中的所有列）。
	输入：
	```python
	arr = np.arange(9).reshape(3,3)
	```
	答案：
	```python
	arr[:,:] = arr[:,: :-1]
	```

20. 如何创建一个包含5和10之间随机浮点的2维数组？（创建一个形态为5×3的2维数组，包含5和10之间的随机十进制小数）。
	答案：
	```python
	np.random.uniform(5, 10, size=(5, 3))
	```

21. 如何在Python NumPy数组中仅输出小数点后三位的数字？（输出或显示NumPy数组rand_arr中小数点后三位的数字）。
	输入：
	```python
	rand_arr = np.random.random((5,3))
	```
	答案：
	```python
	np.random.uniform(5,10,(5,3))
	```

22. 如何通过禁用科学计数法（如1e10）打印 NumPy 数组？（通过禁用科学计数法（如 1e10）打印NumPy数组 rand_arr）。
	输入：
	```python
	# Create the random array
	np.random.seed(100)
	rand_arr = np.random.random([3,3])/1e3
	rand_arr
	#> array([[ 5.434049e-04, 2.783694e-04, 4.245176e-04],
	#> [ 8.447761e-04, 4.718856e-06, 1.215691e-04],
	#> [ 6.707491e-04, 8.258528e-04, 1.367066e-04]])
	```
	期望输出：
	```bash
	#> array([[ 0.000543, 0.000278, 0.000425],
	#> [ 0.000845, 0.000005, 0.000122],
	#> [ 0.000671, 0.000826, 0.000137]])
	```
	答案：
	```python
	np.set_printoptions(suppress=True)
	np.around(rand_arr, decimals=6)
	```

23. 如何限制NumPy数组输出中项的数目？（将Python NumPy数组a输出的项的数目限制在最多6个元素）。
	输入：
	```python
	a = np.arange(15)
	#> array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
	```
	期望输出：
	```bash
	#> array([ 0, 1, 2, ..., 12, 13, 14])
	```
	答案：
	```python
	np.set_printoptions(threshold=6)
	print a
	```

24. 如何在不截断数组的前提下打印出完整的NumPy数组？（在不截断数组的前提下打印出完整的NumPy数组a）。
	输入：
	```python
	np.set_printoptions(threshold=6)
	a = np.arange(15)
	a
	#> array([ 0, 1, 2, ..., 12, 13, 14])
	期望输出：
	```bash
	#> array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
	```
	答案：
	```python
	np.set_printoptions(threshold='nan')
	```

25. 如何向Python NumPy导入包含数字和文本的数据集，同时保持文本不变？（导入iris数据集，保持文本不变）。
	答案：
	```python
	np.genfromtxt('iris.data', delimiter=',', dtype=None)
	```

26. 如何从1维元组数组中提取特定的列？(从前一个问题导入的1维iris中提取文本列species)。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
	```
	答案：
	```python
	np.array([list(i) for i in iris_1d])[:, 4]
	```

27. 如何将1维元组数组转换成2维NumPy数组？(忽略species文本字段，将1维iris转换成2维数组iris_2d)。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
	```
	答案：
	```python
	np.array([list(i) for i in iris_1d])[:, 0:4]
	```

28. 如何计算NumPy数组的平均值、中位数和标准差？（找出iris sepallength（第一列）的平均值、中位数和标准差）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	```
	答案：
	```python
	np.mean(iris[:, 0:4].astype(float)[:, 0])
	```

29. 如何归一化数组，使值的范围在 0 和 1 之间？（创建 iris sepallength 的归一化格式，使其值在 0 到 1 之间。）
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
	```
	答案：
	```python
	def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1 ):
	    def normalizeFunc(x):
	        r=(x-minVal) * newMaxValue / (maxVal-minVal) + newMinValue
	        return r
	    return np.frompyfunc(normalizeFunc, 1, 1)
	normalize_func(np.amin(sepallength), np.amax(sepallength), 0, 1)(sepallength)
	```

30. 如何计算softmax分数？（计算sepallength的softmax分数）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
	```
	答案：
	```python
	np.exp(sepallength - np.max(sepallength)) / np.exp(sepallength - np.max(sepallength)).sum(axis=0)
	```

31. 如何找到NumPy数组的百分数？（找出iris sepallength（第一列）的第5个和第95个百分数）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
	```
	答案：
	```python
	np.percentile(sepallength, 5, axis=0)
	np.percentile(sepallength, 95, axis=0)
	```

32. 如何在数组的随机位置插入值？（在iris_2d数据集中的20个随机位置插入np.nan值）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
	```
	答案：
	```python
	np.insert(iris_2d, np.random.randint(0, iris_2d.shape[0], 20, dtype='int'), values=np.nan, axis=0)
	```

33. 如何在NumPy数组中找出缺失值的位置？（在iris_2d的sepallength（第一列）中找出缺失值的数目和位置）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
	iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
	```
	答案：
	```python
	a = np.isnan(iris_2d[:, 0])
	b = np.full_like(iris_2d[:, 0], True).astype(np.bool)
	np.sum(a)
	np.where(a==b)
	```

34. 如何基于两个或以上条件过滤NumPy数组？（过滤iris_2d中满足petallength（第三列）> 1.5和sepallength（第一列）< 5.0的行）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
	```
	答案：
	```python
	iris_2d[(iris_2d[:, 2] > 1.5) ^ (iris_2d[:, 0] < 5.0)]
	或者
	iris_2d[np.bitwise_xor(iris_2d[:, 2] > 1.5, iris_2d[:, 0] < 5.0)]
	```

35. 如何在NumPy数组中删除包含缺失值的行？（选择iris_2d中不包含nan值的行）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	np.insert(iris_2d, np.random.randint(0, iris_2d.shape[0], 20, dtype='int'), values=np.nan, axis=0)
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
	```
	答案：
	```python
	iris_2d[np.sum(np.isnan(iris_2d), axis = 1) == 0][:5]
	```

36. 如何找出NumPy数组中两列之间的关联性？（找出iris_2d中SepalLength（第一列）和PetalLength（第三列之间的关联性）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
	```
	答案：
	```python
	np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])[0,1]
	```

37. 如何确定给定数组是否有空值？（确定iris_2d是否有缺失值）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
	```
	答案：
	```python
	np.isnan(iris_2d).any()
	```

38. 如何在NumPy数组中将所有缺失值替换成0？（在NumPy数组中将所有nan替换成0）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
	iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
	```
	答案：
	```python
	iris_2d[np.isnan(iris_2d[:, 0]) | np.isnan(iris_2d[:, 1]) | np.isnan(iris_2d[:, 2]) | np.isnan(iris_2d[:, 3])]
	```

39. 如何在NumPy数组中找出唯一值的数量？（在iris的species列中找出唯一值及其数量）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	np.unique(iris.astype(np.str)[:, 4])
	np.sum(iris.astype(np.str)[:, 4] == a[0])
	np.sum(iris.astype(np.str)[:, 4] == a[1])
	np.sum(iris.astype(np.str)[:, 4] == a[2])
	```

40. 如何将一个数值转换为一个类别（文本）数组？将iris_2d的petallength（第三列）转换以构建一个文本数组，按如下规则进行转换：少于3赋值为'small'，3-5赋值为'medium'，大于5赋值为'large'。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	new_iris = np.full_like(iris[:, 2], 'medium')
	new_iris[iris[:,2].astype(np.float) < 3] = 'small'
	new_iris[iris[:,2].astype(np.float) >= 5] = 'large'
	```

41. 如何基于NumPy数组现有列创建一个新的列？为iris_2d中的volume列创建一个新的列，volume指(pi * petallength * sepal_length ^ 2) / 3。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	pi = 3.1415926
	raw_iris = iris[:, 0:4].astype(np.float)
	new_iris = (raw_iris[:, 2] * pi * raw_iris[:, 0] ** 2) / 3
	```

42. 如何在NumPy中执行概率采样？（随机采样iris数据集中的species列，使得setose的数量是versicolor和virginica数量的两倍）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	```
	答案：
	```python
	np.random.seed(100)
	a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
	species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
	```

43. 如何在多维数组中找到一维的第二最大值？（在species setosa的petallength 列中找到第二最大值）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	np.unique(iris[iris[:,4] == "Iris-setosa"][:, 3].astype(np.float))[-2]
	```

44. 如何用给定列将2维数组排序？（基于sepallength列将iris数据集排序）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	iris[iris[:,0].argsort()]
	```

45. 如何在 NumPy 数组中找到最频繁出现的值？（在iris数据集中找到petallength（第三列）中最频繁出现的值）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	答案：
	```python
	vals, counts = np.unique(iris[:, 2], return_counts=True)
	print vals[np.argmax(counts)]
	```

46. 如何找到第一个大于给定值的数的位置？（在iris数据集的petalwidth（第四列）中找到第一个值大于1.0的数的位置）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	```
	答案：
	```python
	np.argmax(iris[:, 3].astype('float') > 1.0)
	```

47. 如何将数组中所有大于给定值的数替换为给定的cutoff值？（对于数组a，将所有大于30的值替换为30，将所有小于10的值替换为10）。
	输入：
	```python
	np.random.seed(100)
	a = np.random.uniform(1,50, 20)
	```
	答案：
	```python
	a[a > 30] = 30
	a[a < 10] = 10
	```

48. 如何在NumPy数组中找到top-n数值的位置？（在给定数组a中找到top-5最大值的位置）。
	输入：
	```python
	np.random.seed(100)
	a = np.random.uniform(1,50, 20)
	```
	答案：
	```python
	np.sort(np.unique(a))[-5:]
	```

49. 如何逐行计算数组中所有值的数量？（逐行计算唯一值的数量）。
	输入：
	```python
	np.random.seed(100)
	arr = np.random.randint(1,11,size=(6, 10))
	#> array([[ 9, 9, 4, 8, 8, 1, 5, 3, 6, 3],
	#> [ 3, 3, 2, 1, 9, 5, 1, 10, 7, 3],
	#> [ 5, 2, 6, 4, 5, 5, 4, 8, 2, 2],
	#> [ 8, 8, 1, 3, 10, 10, 4, 3, 6, 9],
	#> [ 2, 1, 8, 7, 3, 1, 9, 3, 6, 2],
	#> [ 9, 2, 6, 5, 3, 9, 4, 6, 1, 10]])
	```
	期望输出：
	```bash
	#> [[1, 0, 2, 1, 1, 1, 0, 2, 2, 0],
	#> [2, 1, 3, 0, 1, 0, 1, 0, 1, 1],
	#> [0, 3, 0, 2, 3, 1, 0, 1, 0, 0],
	#> [1, 0, 2, 1, 0, 1, 0, 2, 1, 2],
	#> [2, 2, 2, 0, 0, 1, 1, 1, 1, 0],
	#> [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]
	```
	答案：
	```python
	def counts_of_all_values_rowwise(arr2d):
	    # Unique values and its counts row wise
	    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]
	    # Counts of all values row wise
	    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])
	print counts_of_all_values_rowwise(arr)
	```

50. 如何将array_of_arrays转换为平面1维数组？（将array_of_arrays转换为平面线性1维数组）。
	输入：
	```python
	arr1 = np.arange(3)
	arr2 = np.arange(3,7)
	arr3 = np.arange(7,10)
	array_of_arrays = np.array([arr1, arr2, arr3])
	array_of_arrays#> array([array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])], dtype=object)
	```
	期望输出：
	```bash
	#> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	```
	答案：
	```python
	np.concatenate(array_of_arrays)
	```

51. 如何为NumPy数组生成one-hot编码？（计算one-hot编码）。
	输入：
	```python
	np.random.seed(101)
	arr = np.random.randint(1,4, size=6)
	arr
	#> array([2, 3, 2, 2, 2, 1])
	```
	期望输出：
	```bash
	#> array([[ 0., 1., 0.],
	#> [ 0., 0., 1.],
	#> [ 0., 1., 0.],
	#> [ 0., 1., 0.],
	#> [ 0., 1., 0.],
	#> [ 1., 0., 0.]])
	```
	答案：
	```python
	def one_hot_encodings(arr):
	    uniqs = np.unique(arr)
	    out = np.zeros((arr.shape[0], uniqs.shape[0]))
	    for i,k in enumerate(arr):
	        out[i, np.where(uniqs==k)] = 1
	    return out
	one_hot_encodings(arr)
	```

52. 如何创建由类别变量分组确定的一维数值？（创建由类别变量分组的行数。使用以下来自iris species的样本作为输入）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
	species_small = np.sort(np.random.choice(species, size=20))
	species_small
	#> array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
	#> 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
	#> 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
	#> 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
	#> 'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
	#> 'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
	#> dtype='<U15')
	```
	期望输出：
	```bash
	#> [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7]
	```
	答案：
	```python
	[i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])]
	```

53. 如何基于给定的类别变量创建分组id？（基于给定的类别变量创建分组id。使用以下来自iris species的样本作为输入）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
	species_small = np.sort(np.random.choice(species, size=20))
	species_small
	#> array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
	#> 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
	#> 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
	#> 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
	#> 'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
	#> 'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
	#> dtype='<U15')
	```
	期望输出：
	```bash
	#> [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
	```
	答案：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
	np.random.seed(100)
	species_small = np.sort(np.random.choice(species, size=20))
	species_small
	```

54. 如何使用NumPy对数组中的项进行排序？（为给定的数值数组a创建排序）。
	输入：
	```python
	np.random.seed(10)
	a = np.random.randint(20, size=10)
	print(a)
	#> [ 9 4 15 0 17 16 17 8 9 0]
	```
	期望输出：
	```bash
	[4 2 6 0 8 7 9 3 5 1]
	```
	答案：
	```python
	a.argsort().argsort()
	```

55. 如何使用NumPy对多维数组中的项进行排序？（给出一个数值数组a，创建一个形态相同的排序数组）。
	输入：
	```python
	np.random.seed(10)
	a = np.random.randint(20, size=[2,5])
	print(a)
	#> [[ 9 4 15 0 17]
	#> [16 17 8 9 0]]
	```
	期望输出：
	```bash
	#> [[4 2 6 0 8]
	#> [7 9 3 5 1]]
	```
	答案：
	```python
	a.ravel().argsort().argsort().reshape(a.shape)
	```

56. 如何在2维NumPy数组中找到每一行的最大值？（在给定数组中找到每一行的最大值）。
	输入：
	```python
	np.random.seed(100)
	a = np.random.randint(1,10, [5,3])
	#> array([[9, 9, 4],
	#> [8, 8, 1],
	#> [5, 3, 6],
	#> [3, 3, 3],
	#> [2, 1, 9]])
	```
	答案：
	```python
	np.max(a, axis=1)
	```

57. 如何计算2维NumPy数组每一行的最小值除以最大值的结果？（给定一个2维NumPy数组，计算每一行的min-by-max）。
	输入：
	```python
	np.random.seed(100)
	a = np.random.randint(1,10, [5,3])
	a
	#> array([[9, 9, 4],
	#> [8, 8, 1],
	#> [5, 3, 6],
	#> [3, 3, 3],
	#> [2, 1, 9]])
	```
	答案：
	```python
	np.apply_along_axis(lambda x: (np.min(x) * 1.0)/np.max(x), arr=a, axis=1)
	```

58. 如何在NumPy数组中找到重复条目？（在给定的NumPy数组中找到重复条目（从第二次出现开始），并将其标记为True。第一次出现的条目需要标记为False）。
	输入：
	```python
	np.random.seed(100)
	a = np.random.randint(0, 5, 10)
	print('Array: ', a)
	#> Array: [0 0 3 0 2 4 2 2 2 2]
	```
	期望输出：
	```bash
	#> [False True False True False False True True True True]
	```
	答案：
	```python
	out = np.full(a.shape[0], True)
	unique_positions = np.unique(a, return_index=True)[1]
	out[unique_positions] = False
	print out
	```

59. 如何找到NumPy的分组平均值？（在2维NumPy数组的类别列中找到数值的平均值）。
	输入：
	```python
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
	iris = np.genfromtxt(url, delimiter=',', dtype='object')
	names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
	```
	期望输出：
	```bash
	#> [[b'Iris-setosa', 3.418],
	#> [b'Iris-versicolor', 2.770],
	#> [b'Iris-virginica', 2.974]]
	```
	答案：
	```python
	numeric_column = iris[:, 1].astype('float')  # sepalwidth
	grouping_column = iris[:, 4]  # species

	# List comprehension version
	print [[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)]

	# For Loop version
	output = []
	for group_val in np.unique(grouping_column):
	    output.append([group_val, numeric_column[grouping_column==group_val].mean()])
	print output
	```

60. 如何将PIL图像转换成NumPy数组？（从以下URL中导入图像，并将其转换成NumPy数组）。
	https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg
	答案：
	```python
	from io import BytesIO
	from PIL import Image
	import PIL, requests
	# Import image from URL
	URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
	response = requests.get(URL)
	# Read it as Image
	I = Image.open(BytesIO(response.content))
	# Optionally resize
	I = I.resize([150,150])
	# Convert to numpy array
	arr = np.asarray(I)
	```

61. 如何删除NumPy数组中所有的缺失值？（从1维NumPy数组中删除所有的nan值）。
	输入：
	```python
	np.array([1,2,3,np.nan,5,6,7,np.nan])
	```
	期望输出：
	```bash
	array([ 1., 2., 3., 5., 6., 7.])
	```
	答案：
	```python
	a[~np.isnan(a)]
	```

62. 如何计算两个数组之间的欧几里得距离？（计算两个数组a和b之间的欧几里得距离）。
	输入：
	```python
	a = np.array([1,2,3,4,5])
	b = np.array([4,5,6,7,8])
	```
	答案：
	```python
	np.sqrt(np.sum((a-b) ** 2))
	或者
	np.linalg.norm(a-b)
	```

63. 如何在一个1维数组中找到所有的局部极大值（peak）？（在1维数组a中找到所有的peak，peak指一个数字比两侧的数字都大）。
	输入：
	```python
	a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
	```
	期望输出：
	```bash
	#> array([2, 5])
	```
	答案：
	```python
	np.array([i for i in range(1, a.shape[0]-1) if a[i] > a[i-1] and a[i] > a[i+1]])
	```

64. 如何从2维数组中减去1维数组，从2维数组的每一行分别减去1维数组的每一项？（从2维数组a_2d中减去1维数组b_1d，即从a_2d的每一行分别减去b_1d）。
	输入：
	```python
	a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
	b_1d = np.array([1,1,1])
	```
	期望输出：
	```bash
	#> [[2 2 2]
	#> [2 2 2]
	#> [2 2 2]]
	```
	答案：
	```python
	a_2d-b_1d
	```

65. 如何在数组中找出某个项的第n个重复索引？（找到数组x中数字1的第5个重复索引）。
	输入：
	```python
	x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
	```
	答案：
	```python
	[i for i, v in enumerate(x) if v == 1][n-1]
	或者
	np.where(x == 1)[0][n-1]
	```

66. 如何将NumPy的datetime64对象（object）转换为datetime的datetime对象？（将NumPy的datetime64对象（object）转换为datetime的datetime对象）。
	输入：
	```python
	dt64 = np.datetime64('2018-02-25 22:10:10')
	```
	答案：
	```python
	from datetime import datetime
	datetime.utcfromtimestamp((dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
	```

67. 如何计算NumPy数组的移动平均数？（给定1维数组，计算window size为3的移动平均数）。
	输入：
	```python
	np.random.seed(100)
	Z = np.random.randint(10, size=10)
	```
	答案：
	```python
	def moving_average(a, n=3):
	    ret = np.cumsum(a, dtype=float)
	    print ret[n:]
	    ret[n:] = ret[n:] - ret[:-n]
	    return ret[n - 1:] / n
	print moving_average(Z)
	或者
	np.convolve(Z, np.ones(3)/3, mode='valid')
	```

68. 给定起始数字、length和步长，如何创建一个NumPy数组序列？（从5开始，创建一个length为10的NumPy数组，相邻数字的差是3）。
	答案：
	```python
	np.arange(10) * 3 + 5
	```

69. 如何在不规则NumPy日期序列中填充缺失日期？（给定一个非连续日期序列的数组，通过填充缺失的日期，使其变成连续的日期序列）。
	答案：
	```python
	np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
	```

70. 如何基于给定的1维数组创建 strides？（给定1维数组arr，使用strides生成一个2维矩阵，其中window length等于4，strides等于2）。
	[[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
	输入：
	```python
	arr = np.arange(15)
	arr
	#> array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
	```
	期望输出：
	```bash
	#> [[ 0 1 2 3]
	#> [ 2 3 4 5]
	#> [ 4 5 6 7]
	#> [ 6 7 8 9]
	#> [ 8 9 10 11]
	#> [10 11 12 13]]
	```
	答案：
	```python
	def gen_strides(arr, stride_len=5, window_len=5):
	    n_strides = ((a.size-window_len)//stride_len) + 1
	    # return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
	    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])
	print(gen_strides(np.arange(15), stride_len=2, window_len=4))
	```
