# !usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np 
import pandas as pd

from KNN_core import auoNorm
from KNN_core import classfy0

"""
对于约会数据的项目
"""

def datingClasfy():
    """
    Desc:
        对约会进行数据分析
    Args:
        None
    Return:
        None
    """
    # 测试集的比例
    txtFile="AiLearning_xqf\\1_KNN\\data\\datingTestSet2.txt"
    hoRatio=0.1 
    # 加载数据
    datingDataMatrix,datingDataLabels=file2Matrix(txtFile)
    # 归一化数据,包括测试数据
    normDataSet,ranges,minVals=auoNorm(datingDataMatrix)
    m=normDataSet.shape[0]
    numTestVec=int(m*hoRatio)
    print("测试数据集的行数：",numTestVec)
    error=0.0
    for i in range(numTestVec):
        # 每一个测试数据都是和全量训练数据比对
        classfyResult=classfy0(normDataSet[i],normDataSet[numTestVec:m],datingDataLabels[numTestVec:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classfyResult, datingDataLabels[i]))
        if (classfyResult != datingDataLabels[i]): error += 1.0
    print("the total error rate is: %f" % (error / float(numTestVec)))
    print(numTestVec)



def file2Matrix(filepath):
    """
    Desc:
        把文本文件解析成我想要的矩阵训练集和标签，但是这里比较奇怪的是txt文件，我想要用pandas重写一遍
    Args:
        filepath:文件路径
    Returns:
        returnMatrix:训练集数据
        classLabelVector：结果标签
    """
    # df=pd.read_csv(filepath,sep=' ')还真不能这样用，因为好像也不是简简单单的空格，空格包含的元素个数还不一样、
    # 只能是一行一行的读了
    # 为什么不能两次的file.readlines()
    # with open(filepath) as file:
    #     resultlines=file.readlines()
    #     numLines=len(resultlines)
    #     # 先把位置站好，开坑
    #     # returnMatrix=np.zeros(numLines,3)  报错
    #     returnMatrix=np.zeros((numLines,3)) 
    #     classLabelVector=[]
    #     index=0
    #     for line in resultlines:
    #         line=line.strip()
    #         listFromLine=line.split("\t")
    #         returnMatrix[index]=listFromLine[:3]
    #         classLabelVector.append(int(listFromLine[-1]))
    #         index+=1

    # 使用pd,完美实现功能，但是为什么打开txt文件看到里面并不像制表符啊
    df=pd.read_csv(filepath,sep="\t")
    returnMatrix=df.iloc[:,0:3]
    returnMatrix=np.array(returnMatrix)
    classLabelVector=df.iloc[:,3]
    classLabelVector=np.array(classLabelVector)

    # 果然是这样，只用dataframe的话连类型都不一样，classfy0接受的参数是ndarray类型。所以先要进行一个类型转换

    # numpy中的ndarray与pandas的Series和DataFrame之间的相互转换 https://blog.csdn.net/jinguangliu/article/details/78538748

    return returnMatrix,classLabelVector


if __name__ == "__main__":
    datingClasfy()