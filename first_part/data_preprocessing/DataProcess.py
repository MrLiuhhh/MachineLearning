# -*- coding: utf-8 -*-:
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

# pandas 读取数据
test = pd.read_csv('../data/train.csv', header=0)
id=test.get('Id')
# print(id)

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
