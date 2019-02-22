# -*- coding: utf-8 -*-:
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder


# 显示所有行
# pd.set_option('display.max_rows', None)
# 显示所有列
pd.set_option('display.max_columns', None)

# pandas 读取数据
test_csv = pd.read_csv('../data/train.csv', header=0)
# 获取列标签
head_value = test_csv.columns.values
# print(test_csv)

# Imputer 填补数据
# missing_values 设置缺失值，可以为整数或者NaN
# strategy 替换缺失值的策略： 1、mean,用特征列的均值替换 2、median，用特征列的中位数数进行替换  3、most_frequent,用特征列的众数替换
# axis 指定轴数， 0:表示选取的是列，1:表示的是行 ,默认为0
# copy :True 表示不在原数据集修改  ，False 表示在原数据集进行修改; 存在如下情况时，即使设置为False时，也不会就地修改: a、不是浮点值数组
# b、X是稀疏且missing_values=0   c、axis=0且X为CRS矩阵   d、axis=1且X为CSC矩阵
test_array = np.array([[1.0, 2.0],
                       [np.nan, 3.0],
                       [4.0, 5.0]])
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy='False')
imp.fit(test_array)
print(test_array)
x = imp.transform(test_array)
print(imp.transform(x))

# LabelEncoder 处理类别型数据
# LabelEncoder是对不连续的数字或者英文等从1～n进行编号，比如这里有一批标签｛A,A,B,B,C,D,D｝
# 那么这里标签的种类就有 ：A  B  C  D,且是根据字典排序的，那么相应的标签对应的序号便是 0，1，2，3
# 所以同理｛D，A,A,C,B｝=> {3,0,0,2,1}
label = LabelEncoder()
test_label2 = test_csv.get('MSZoning')
print("标签值：%s" % test_label2)
label_result = label.fit(test_label2)
result = label_result.transform(test_label2)
print("标签 %s:" % label.classes_)
print("标准话后：%s" % result)


