#!/usr/bin/python
#-*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as ai
import numpy as np
import time

def loadData():    #加载数据函数
    dataMat = []
    labelMat1 = []
    labelMat2 = []
    labelMat3 = []
    ylabel = []
    fr = open('iris.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        if(lineArr[4]=='Iris-setosa'):
          labelMat1.append(float(1))
        else:
          labelMat1.append(float(-1))
        if(lineArr[4]=='Iris-versicolor'):
          labelMat2.append(float(1))
        else:
          labelMat2.append(float(-1))
        if(lineArr[4]=='Iris-virginica'):
          labelMat3.append(float(1))
        else:
          labelMat3.append(float(-1))
        ylabel.append(lineArr[4])
    return dataMat,labelMat1,labelMat2,labelMat3,ylabel


def pca(dataMat, topNfeat):              #pca降维
    re = dataMat - mean(dataMat, axis = 0)   #去除平均值 
    covMat = cov(re,rowvar=0) #计算协防差矩阵  
    eVals, eVects = linalg.eig(mat(covMat))  
    eValInd = argsort(eVals)  
    #从小到大对N个值排序  
    eValInd = eValInd[: -(topNfeat + 1) : -1]  
    reVects = eVects[:, eValInd]
    lowDataMat = re * reVects   #转换到降维空间
    return lowDataMat

def selectJrand(i,m):                         #随机选择alpha
    j=i             #排除i
    while (j==i):
          j = int(random.uniform(0,m))
    return j
     
def clipAlpha(aj,H,L):                       #规范alpha的值
    if aj > H:
       aj = H
    if L > aj:
       aj = L
    return aj

def smoSimple(dataMatrix, classLabels, C, toler, maxIter):       #简单SMO算法求解
    labelMat = mat(classLabels).T
    b = -1; m,n = shape(dataMatrix) 
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0   #alpha是否已经进行了优化
        for i in range(m):
            #   w = alpha * y * x;  f(x_i) = w^T * x_i + b
            # 预测的类别
            fXi = float(multiply(alphas,labelMat).T*dataMatrix*dataMatrix[i,:].T) + b    
            Ei = fXi - float(labelMat[i])   #得到误差，如果误差太大，检查是否可能被优化
            #必须满足约束
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): 
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()                
                if (labelMat[i] != labelMat[j]):                                          
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                   continue
                # Eta = -(2 * K12 - K11 - K22)，且Eta非负，此处eta = -Eta则非正
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                   continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                  #如果内层循环通过以上方法选择的α_2不能使目标函数有足够的下降，那么放弃α_1
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
    return b,alphas

def calcWs(alphas,dataMatrix, labelMat):      #求出参数w
    m,n = shape(dataMatrix) 
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],dataMatrix[i,:].T)
    return w


def show(dataxArr, ydata,b1, b3):           #显示函数
    n = len(dataxArr)
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    xcord3 = [];ycord3 = []
    x = arange(-4,5,0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dataxArr = mat(dataxArr)
  
    #原始数据
    for i in range(n):
        if ydata[i]=='Iris-setosa':
            xcord1.append(dataxArr[i,0]);ycord1.append(dataxArr[i,1])
        elif ydata[i]=='Iris-versicolor':
            xcord2.append(dataxArr[i,0]);ycord2.append(dataxArr[i,1])
        else:
            xcord3.append(dataxArr[i,0]);ycord3.append(dataxArr[i,1])
    
    #分类边界
    xx,yy= np.meshgrid(x,x)
    z = xx * w1[0][0] + yy * w1[1][0] + float(b1)
    z1 = xx * w3[0][0] + yy * w3[1][0] + float(b3)
    z.reshape(xx.shape)
    z1.reshape(xx.shape)
    m ,n = shape(z)
    for i in range(m):
        for j in range(n):
            if(z[i][j]>0):
               z[i][j]=1
            elif(z1[i][j]>0):
               z[i][j]=26
            else:
               z[i][j]=50

    #图形的显示
    plt.contourf(xx, yy, z, 10, cmap = plt.cm.Spectral)
    ax.scatter(xcord1,ycord1,s=30,c='b',marker='o',label="Iris-setosa")
    ax.scatter(xcord2,ycord2,s=30,c='g',marker='o',label="Iris-versicolor")
    ax.scatter(xcord3,ycord3,s=30,c='r',marker='s',label='Iris-virginica')
    plt.xlabel('X1',size=25)                  # 横坐标
    plt.ylabel('X2',size=25)                  # 纵坐标
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title ('SVM_PCA',size=30)             # 标题
    plt.xticks(())
    plt.yticks(())
    plt.show()  
  
xdata,ydata1,ydata2,ydata3,ylabe = loadData() #加载数据
X = pca(xdata,2) #将数据降为2维
b1 , alphas1 = smoSimple(X,ydata1,0.8,0.0001,40)  #求解第一个分隔平面的参数
b3 , alphas3 = smoSimple(X,ydata3,0.8,0.0001,40)  #求解第二个分隔平面的参数
w1 = calcWs(alphas1,X,ydata1) #求解w1
w3 = calcWs(alphas3,X,ydata3) #求解w3
show(X, ylabe,b1,b3)  #显示图形
