# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:25:29 2019

@author: matt
"""
import numpy as np
class logisticRegression(object):
	def __init__(self):
		self.w = None

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def train(self,X,Y,learningRate=1e-3,numIters=1000,reg=0.0,verbose=False):
		m,n = X.shape
		n = np.shape(X)[1]   #特征个数
		#权值初始化
		if self.w == None:
			self.w = np.ones((n,1))
		lossHistory = []
		#批量梯度下降
		for i in range(numIters):
			h = self.sigmoid(X.dot(self.w))
			loss = h-Y
			lossHistory.append(np.sum(loss))
			#更新参数
			self.w = self.w - learningRate*X.T.dot(loss)
			if verbose and i%100 == 0:
				print("迭代次数:%d/%d  loss: %f"%(i,numIters,np.sum(loss)))
		return lossHistory

	def predict(self,X):
		return np.mat([1 if x >= 0.5 else 0 for x in self.sigmoid(X.dot(self.w))])












