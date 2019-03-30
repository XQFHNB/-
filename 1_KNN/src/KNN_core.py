# !usr/bin/env python3
# -*- coding:utf-8 -*-

"""
核心：就是已经有一堆已知标签得数据集，然后把未知的数据放进去与各个已知的数据集进行比较，选最近k个数据点类别最多的
（我这好像是写的人家的通俗理解）
"""
import numpy as np 
import pandas as pd
# from numpy import *
import operator 
import os 
from collections import Counter


def classfy0(inX,dataSet,labels,k):
    """
    Desc:
        KNN的分类函数
    Args:
        inX:想要被分类的输入向量
        dataSet:训练数据集的features
        labels:训练数据集的labels
        k:选择最近邻的数目
    Returns:
        sortedClassCount[0][0]  结果标签类别

    !! labels行数与dataSet的行数相同，程序使用欧式距离公式
    """

    # 1 距离计算
    dataSetSize=dataSet.shape[0]
    # 把inx向量重复dataSetSize次再减去dataSet，生成结果矩阵
    diffMatrix=np.tile(inX,(dataSetSize,1))-dataSet
    # 平方一下
    squareDiffMatrix=diffMatrix**2
    # 行加和
    sumDistances=squareDiffMatrix.sum(axis=1)
    #  开方
    distances=sumDistances**0.5
    # 其实到这里这一个输入向量与所有点的欧式距离都是已经算出来了，接下来就是进行排序了
    # 终于理解了为什么要使用argsort()而不直接使用sort()排一下序了，因为我要保持索引位置才能在labels里面找到对应的label
    sortedDistanceDices=distances.argsort()  #返回排序后的索引，然后就可以拿着索引去label里面挑出label了

    # 2 选择距离最小的k个点
    # 还是老办法 使用字典来作为计数容器

    classCount={}
    for i in range(k):
        label=labels[sortedDistanceDices[i]]
        # 多个点可能属于同一个label
        classCount[label]=classCount.get(label,0)+1         

    # 得到一个记录了label出现次数的字典，然后选出字典中value值最大的key(label)就成功了
    # 对字典的值进行逆序，排序的依据是比较第二个元素，也就是值
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) 
    return sortedClassCount[0][0]  

def classfy1(inX,dataSet,labels,k):
    """
    Desc:
        KNN的分类函数
    Args:
        inX:想要被分类的输入向量
        dataSet:训练数据集的features
        labels:训练数据集的labels
        k:选择最近邻的数目
    Returns:

    !! labels行数与dataSet的行数相同，程序使用欧式距离公式
    """
    # 计算距离
    dist=np.sum((inX-dataSet)**2,axis=1)**0.5
    # 获取前k个label，我只需要前k个就可以了
    k_labels=[labels[index] for index in dist.argsort()[0:k]]
    # 计算出现次数最多
    label=Counter(k_labels).most_common(1)[0][0]
    return label


def auoNorm(dataSet):
    """
    Desc:
        归一化特征值
    Args:
        dataSet:需要进行归一化的数据集
    Returns:
        normDataSet:进行归一化后的数据集
        ranges:归一化处理的范围
        minVal:最小值
    归一化公式：Y=(X-Xmin)/(Xmax-Xmin)
    其中Xmin和Xmax分别为最小和最大的特征值，该函数可自动将数字特征转化为0-1之间
    """
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    # 应该是相同的
    # normDataSet=np.zeros(np.shape(dataSet))
    # normDataSet=np.zeros(dataSet.shape)

    normDataSet=(dataSet-minVals)/ranges
    return normDataSet,ranges,minVals


def createDataSet():
    """
    Desc:
        创建数据集和标签
    Args:
        None
    Returns:
        group:训练数据集的features
        labels:训练数据集的labels
    """
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels



def testClassfy0():
    group,labels=createDataSet()
    print(str(group))
    print(str(labels))
    print(classfy0([0.1,0.1],group,labels,3))

if __name__ == "__main__":
    print("===========")
    # testClassfy0()
    # print(file2Matrix(txtFile))





