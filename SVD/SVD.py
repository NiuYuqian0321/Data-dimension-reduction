#coding=utf-8
import numpy  as np
import operator

def load_dataset(file_name, delimiter='\t'):
	dataArr = []
	labelArr = []
	with open(file_name) as f:
		for line in f:
			stringLine = line.strip().split(delimiter)
			dataLine = list(map(float, stringLine))
			labelArr.append(dataLine.pop())
			dataArr.append(dataLine)
	return np.mat(dataArr), labelArr


def svd(dataset, K=99999):
    #对所有样本中心化
	mean_ = np.mean(dataset, axis=0)
	dataset = dataset - mean_
	#对样本进行奇异值分解
	u, sigma, vT = np.linalg.svd(dataset)
	#降到K维，取前K个
	#new_u = u[:, :K]
	#new_sigma = sigma[0:K]
	new_vT = vT[:K, :]
    #得出投影矩阵，即降维后的v
	W = new_vT.T
	return W
