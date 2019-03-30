# !usr/bin/env python3
# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
import os

from KNN_core import classfy0
from KNN_core import auoNorm


def img2Vector(filePath):
    """
    Desc:
        将图像矩阵转换成向量
    Args:
        filePath:图像存在位置
    Returns:
        returnVector:图像变成的向量
    """
    returnVectorListStr = ''
    with open(filePath) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            returnVectorListStr += str(line).replace("[", '').replace(']', '')
        returnVector = np.array(list(returnVectorListStr))
        returnVector = returnVector.astype(np.int8)
    return returnVector


# def img2Vector(filePath):
#     returnVector=np.zeros(1,1024)
#     fr=open(filePath)
#     for i in range(32):
#         line=fr.readline()
#         for j in line:
#             returnVector[1,32*i+j]=int(line[j])
#     return returnVector


def importData(fileRoot):
    """
    Desc:
        通用的导入数据的方法
    Args:
        fileRoot:文件目录
    Returns:
        dataSet:结果数据集
        dataSet:结果数据集标签
    """
    labels = []
    dataFileList = os.listdir(fileRoot)
    m = len(dataFileList)
    print("文件数量：", m)
    dataSet = np.zeros((m, 1024))
    index = 0
    for fileName in dataFileList:
        labels.append(fileName.split('.')[0].split("_")[0])
        # 组合路径还有这种写法，见识了
        dataSet[index] = img2Vector(fileRoot+"\\"+fileName)
        index += 1
    return dataSet, labels


def handWritingClassfy():
    """
    Desc:
        手写数字识别器，将分类错误数和错误率打印出来
    Args:
        None
    Returns:
        None
    """
    # 1 导入训练数据
    trainDataRoot = 'AiLearning_xqf\\1_KNN\data\\trainingDigits'
    trainSet, trainLabels = importData(trainDataRoot)

    # 2 导入测试数据
    testDataRoot = "AiLearning_xqf\\1_KNN\data\\testDigits"
    testSet, testLabels = importData(testDataRoot)

    testNum = testSet.shape[0]
    error = 0
    for i in range(testNum):
        test = testSet[i]
        classfyResult = classfy0(test, trainSet, trainLabels, 3)
        if classfyResult != testLabels[i]:
            error += 1
    print("错误率：%f" % float(error/testNum))

if __name__ == "__main__":
    filePath = "AiLearning_xqf\\1_KNN\data\\trainingDigits\\0_0.txt"
    # print(img2Vector(filePath))
    # print(img2Vector(filePath)[:10])
    handWritingClassfy()