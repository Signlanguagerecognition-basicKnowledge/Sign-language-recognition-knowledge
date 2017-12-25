# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:03:39 2017

@author: Wuyifan

学习目标：
1.了解到pytorch是numpy的替代品，它的变量支持GPU计算
2.pytorch是一个用于研究的神经网络平台，它的神经网络变量值是可修改的
"""

from __future__ import print_function
import torch
import numpy as np

'''
初始化
'''
#用0初始化tensor
x = torch.Tensor(5, 3)
print(x)

#随机初始化tensor
x = torch.rand(5,3)
#print(x)

#获得tensor的大小 
#size是一个tuple
#print(x.size())

'''
基本运算
'''

#两种形式的加法
y=torch.rand(5,3)
#print(torch.add(x,y))
#print(x+y)

result=torch.Tensor(5,3)
torch.add(x,y,out=result)
#print(result)

#print(y.add_(x))

#切片
#print(y[:1])

'''
numpy和pytorch的变量相互转换
'''

#从torch转化为numpy
a=torch.ones(5,3)
b=a.numpy()
#print(a)
#print(b)
a.add_(1)
#print(a)
#print(b)

#从numpy转化为torch
a=np.ones(5)
b=torch.from_numpy(a)
#print(a)
#print(b)
np.add(a,1,out=a)
#print(a)

'''
把tensor放入到cuda中
'''
if torch.cuda.is_available():
    x=x.cuda()
    y=y.cuda()
    print(x+y)

