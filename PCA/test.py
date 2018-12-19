#coding=utf-8
import numpy  as np
import operator

from PCA import load_dataset, pca

def transform(dataset, W):
	#对所有样本中心化
	mean_ = np.mean(dataset, axis=0)
	dataset = dataset - mean_

	return dataset * W


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
	#print(count)
	accuracy = float(count)/len(dataset_test)
	return accuracy


if __name__=='__main__':
	#加载数据集
	dataset_train, label_train = load_dataset('../two datasets/sonar-train.txt', ',')
	dataset_test, label_test = load_dataset('../two datasets/sonar-test.txt', ',')
	#根据训练集计算投影矩阵
	K = [10, 20, 30]
	for k in K:	
		W = pca(dataset_train, k)
		
		#计算降维后的样本
		dataset_train_K = transform(dataset_train, W) 
		dataset_test_K = transform(dataset_test, W)

		#1NN方法计算准确率
		accuracy = oneNN(dataset_train_K, dataset_test_K, label_train, label_test);
		print "k= %d, accuracy= %f" % (k, accuracy) 

