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


def pca(dataset, K=99999):
    #对所有样本中心化
    mean_ = np.mean(dataset, axis=0)
    dataset = dataset - mean_
	#计算样本的协方差矩阵
    cov_matrix = np.cov(dataset, rowvar=0)
    #print(cov_matrix)
	#对协方差矩阵做特征值分解
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_matrix))
    #print(eig_values)
    eig_index = np.argsort(eig_values)
    #print(eig_index)
	#取最大的K个特征值所对应的特征向量
    eig_index = eig_index[:-(K+1):-1]
	#eig_index = eig_index[-(K):]
    #得出投影矩阵
    W = eig_vectors[:, eig_index]
    return W
