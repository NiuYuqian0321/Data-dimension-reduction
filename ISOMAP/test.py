#coding=utf-8
import numpy  as np
import operator
import time

from ISOMAP import load_dataset, isomap

def transform_MDS(D, K):
	N = D.shape[0]
	B = np.zeros((N,N))
	#根据矩阵D计算内积矩阵B
	#如果不连通，即距离无穷大，则将距离设置为最大距离+1
	maxDist=-float("inf")
	for i in range (N):
		for j in range(i, N):
			if D[i,j] > maxDist and D[i,j] != float("inf"):
				maxDist = D[i,j]

	for i in range (N):
		for j in range(N):
			if D[i,j] == float("inf"):
				D[i,j] = maxDist + 1
	#计算dist..
	#print("D**2")
	#print(D**2)
	ss = 1.0/N**2*np.sum(D**2)

	for i in range(N):
		for j in range(i,N):
			#计算b_ij = -0.5(dist_ij^2 - dist_i.^2 - dist_.j^2 + dist_..^2)
			B[i,j] = B[j,i] = -0.5*(D[i,j]**2 -1.0/N*np.dot(D[i,:],D[i,:].T) -1.0/N*np.dot(D[:,j].T,D[:,j])+ss)

	#对矩阵B进行特征值分解,得到特征值和特征向量
	eig_values, eig_vectors = np.linalg.eig(np.mat(B))
	#取最大K个特征值和特征向量
	eig_index = np.argsort(eig_values)
	eig_index = eig_index[:-(K+1):-1]
	eig_vectors = eig_vectors[:, eig_index]
	eig_values.sort()
	
	eig_values = eig_values[:-(K+1):-1]
	#降维后样本 = 前K个对应的特征向量矩阵 × sqrt(特征值对角矩阵)
	#这里，Z被认为已经中心化
	Z = np.dot(eig_vectors, np.diag(np.sqrt(eig_values)))
	return Z


def oneNN(dataset_train, dataset_test, label_train, label_test):
	#mat转换为array
	dataset_train = np.array(dataset_train)	
	dataset_test = np.array(dataset_test)
	
	#训练集数量（行数）
	numSamples = dataset_train.shape[0]

	#测试结果正确的数目
	count = 0

	#测试集当前行号
	index_test = 0

	#从测试集中取一行，分别和训练集中的每一行计算欧氏距离
	for data_test in dataset_test:
		diff = np.tile(data_test, (numSamples, 1)) - dataset_train
		squaredDiff = diff ** 2
		squaredDist = squaredDiff.sum(axis = 1)
		distance = squaredDist ** 0.5
		#对每一行计算出来的距离进行从小到大排序
		sortedDistIndices = np.argsort(distance)
		#距离最小的行标对应的测试集中的label为测试结果
		index_vote = sortedDistIndices[0]
		label_vote = label_train[index_vote]
		label_true = label_test[index_test]
		#计算准确率	
		if label_vote == label_true:
			count = count + 1
		
		index_test = index_test + 1
	accuracy = float(count)/len(dataset_test)
	return accuracy




if __name__=='__main__':
	#加载数据集
	dataset_train, label_train = load_dataset('../two datasets/sonar-train.txt', ',')
	dataset_test, label_test = load_dataset('../two datasets/sonar-test.txt', ',')

	#根据训练集和测试集，用X-NN构建权重图，得出距离矩阵D
	X_train = 13
	X_test = 13

	print ("train-%dNN:" % (X_train))
	print ("test-%dNN:" % (X_test))
	D_train = isomap(dataset_train,X_train)
	D_test = isomap(dataset_test,X_test)
	#距离矩阵D作为MDS算法输入，计算K维空间的降维后的样本
	K = [10, 20, 30]
	for k in K:	
		#计算降维后的样本
		dataset_train_K = transform_MDS(D_train, k) 
		dataset_test_K = transform_MDS(D_test, k)

		#1NN方法计算准确率
		accuracy = oneNN(dataset_train_K, dataset_test_K, label_train, label_test);
		print "k = %d, accuracy= %f" % (k, accuracy) 

