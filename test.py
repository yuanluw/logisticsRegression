# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:48:06 2019

@author: matt
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from logistic import *

def load_data():

	data = datasets.load_iris()
	offset = np.ones(100)
	feature_data = data['data'][:100]
	feature_data = np.insert(feature_data, 0, values=offset, axis=1)
	label_data = data['target'][:100].reshape((-1,1))

	#80个数据作为训练集 20个作为测试集
	trainX = feature_data[:40]
	trainX = np.concatenate((trainX, feature_data[50:90]))
	trainY = label_data[:40]
	trainY = np.concatenate((trainY, label_data[50:90]))

	testX = feature_data[40:50]
	testX = np.concatenate((testX, feature_data[90:100]))
	testY = label_data[40:50]
	testY = np.concatenate((testY, label_data[90:100]))

	return trainX,trainY,testX,testY


if __name__ == '__main__':
	print("-----------1.load_data-----------")
	trainX,trainY,testX,testY = load_data()
	print("-----------2.training-----------")
	model = logisticRegression()
	lossHistory = model.train(trainX,trainY,learningRate=1e-3,verbose=True,numIters=500)
	ax = plt.subplot(111)
	ax.plot(lossHistory)
	ax.set_xlabel('Iterations ')
	ax.set_ylabel('Cost')
	#验证集测试
	predY = model.predict(testX)
	print("验证集正确率: %f"%((predY.T==testY).mean()))









