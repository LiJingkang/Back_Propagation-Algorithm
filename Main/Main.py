#encoding=utf-8
import numpy as np
import copy
import loadData
def nonLin(x,deriv = False):
    '''''
    计算sigmoid函数，可以返回其导数值
    :param x:输入向量或者标量
    :param deriv: 是否求导
    :return: 返回向量或者标量
    '''
    if deriv == True:
        return x*(1-x)
    return 1.0/(1.0+np.exp(-x))


def initParameter(featureNum):
    '''''
    初始化单层神经网络参数，默认只有一个输出单元
    :param featureNum: 特征数
    :return: 单层神经网络权重向量和阈值
    '''
    np.random.seed(1)
    #生成（特征数*1）的随机浮点数向量，均值为0，列值为1的原因为只有一个输出单元
    syn0 = 2*np.random.random((featureNum,1))-1
    theta = 2*np.random.random()-1 # 在单层网络中，最后只有一个输出单元，因此也就只有一个1*1的阈值
    return syn0,theta


def train1(dataSet,labelsSet,iterNum,alpha):
    '''''
    一层神经网络训练
    :param dataSet: 数据集，列表形式
    :param labelsSet: 标签集，列表形式
    :param iterNum: 最大迭代次数
    :param alpha: 学习率
    :return: 学习后的权重向量和阈值
    '''
    X = np.array(dataSet)
    Y = np.array(np.mat(labelsSet).T)
    featureNum = X.shape[1]
    syn0,theta = initParameter(featureNum)
    for iter in xrange(iterNum):
        l0 = X
        l1 = nonLin(np.dot(l0,syn0)-theta) # 向前传播
        l1Error = Y - l1
        l1Delta = l1Error*nonLin(l1,True) # 数组相乘，对应元素相乘
        syn0 += np.dot(l0.T,alpha*l1Delta) # 更新权重，这里的alpha乘在哪里待定，数据集转置的原因是公式中每个输入分量都要乘以残差项，最后加和一起更新权重
        theta -= sum(alpha*l1Delta) # 由于之前均为批量操作，而在这里阈值是一个1*1的数值，因此用加和的形式做批量处理
    return syn0,theta


def initParameter2(featureNum,hiddenNum):
    '''''
    两层神经网络参数的初始化
    :param featureNum: 特征数目
    :param hiddenNum: 隐藏层数目
    :return: 初始化的两层权重向量和阈值向量
    '''
    syn0 = 2*np.random.random((featureNum,hiddenNum))-1 # 维度为（特征数*隐藏单元数）
    syn1 = 2*np.random.random((hiddenNum,1))-1 # 维度为（隐藏单元数*1）
    theta1 = 2*np.random.random((1,hiddenNum))-1 # 维度为（1*隐藏单元数）
    theta2 = 2*np.random.random()-1 # 这里只有一个输出单元，因此这里仍是一个1*1的数值
    return syn0,syn1,theta1,theta2


def train2(dataSet,labelsSet,iterNum,alpha,hiddenNum):
    '''''
    两层神经网络训练
    :param dataSet:数据集，列表形式
    :param labelsSet: 标签集，列表形式
    :param iterNum: 最大迭代次数
    :param alpha: 学习率
    :param hiddenNum: 隐藏层数目
    :return: 学习后的两层权重向量和阈值向量
    '''
    #X = loadData.normalize(dataSet)
    X = np.array(dataSet)
    Y = np.array(np.mat(labelsSet).T)
    featurNum = X.shape[1]
    sampleNum = X.shape[0]
    syn0,syn1,theta0,theta1 = initParameter2(featurNum,hiddenNum)
    for iter in xrange(iterNum):
        l0 = X
        l1 = nonLin(np.dot(l0,syn0)-np.tile(theta0,[sampleNum,1]))# 向前传播，注意这里的的复制矩阵的操作类似于广播的操作
        l2 = nonLin(np.dot(l1,syn1)-theta1)

        l2Error = Y - l2
        l2Delta = l2Error*nonLin(l2,True)
        oldSyn1 = copy.deepcopy(syn1) # 保留一下之前的权重，以便于第一层的更新
        syn1 += l1.T.dot(l2Delta*alpha) # 更新权重
        theta1 -= sum(alpha*l2Delta) # 更新阈值

        l1Error = l2Delta.dot(oldSyn1.T) # 注意这里前一层误差是由后一层计算得出的
        l1Delta = l1Error*nonLin(l1,True)
        syn0 += l0.T.dot(l1Delta*alpha)
        theta0 -= np.sum(l1Delta,0)*alpha # 按列求和，theta0为隐藏层的权值，维度为（1*隐藏层）
    return syn0,syn1,theta0,theta1

def test1(dataSet,labelsSet,syn0,theta):
    '''''
    测试一层神经网络函数
    :param dataSet: 数据集，列表形式
    :param labelsSet: 标签集，列表形式
    :param syn0: 学习后的权重向量
    :param theta: 学习后的阈值
    :return: 无
    '''
    X = np.array(dataSet)
    Y = np.array(np.mat(labelsSet).T)
    l0 = X
    l1 = nonLin(np.dot(l0,syn0)-theta)
    ans = l1 > 0.5
    res = ans == Y
    print sum(res)
    print sum(res)/float(X.shape[0])
    #print ans
    #print l1

def test(dataSet,labelsSet,syn0,syn1,theta0,theta1):
    '''''
    测试两层神经网络函数
    :param dataSet: 数据集，列表形式
    :param labelsSet: 标签集，列表形式
    :param syn0: 第一层权重
    :param syn1: 第二层权重
    :param theta0: 第一层阈值向量
    :param theta1: 第二层阈值
    :return: 无
    '''
    X = np.array(dataSet)
    Y = np.array(np.mat(labelsSet).T)
    l0 = X
    l1 = nonLin(np.dot(l0,syn0)-theta0)
    l2 = nonLin(np.dot(l1,syn1)-theta1)
    ans = l2 > 0.5
    res = ans == Y
    print sum(res)
    print sum(res)/float(X.shape[0])
    #print ans
    #print l2